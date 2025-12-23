import dataclasses
import itertools
from typing import Callable, NamedTuple
import functools
import jax
from jax import jit
import jax.numpy as jnp
import numpy as np

from slaterform.basis import basis_block
from slaterform.basis import broadcasting
from slaterform.integrals import gaussian

# A one-electron operator between two BasisBlocks.
# The returned array has shape (d1+1, d1+1, d1+1, d2+1, d2+1, d2+1)
# where d1 = basis1.max_degree and d2 = basis2.max_degree.
OneElectronOperator = Callable[
    [gaussian.GaussianBasis3d, gaussian.GaussianBasis3d], np.ndarray
]

# A two-electron operator between four BasisBlocks.
# The returned array has shape
# (d1+1, d1+1, d1+1,
#  d2+1, d2+1, d2+1,
#  d3+1, d3+1, d3+1,
#  d4+1, d4+1, d4+1)
# where di = the max_degree in basisi.cartesian_powers for i = 1, 2, 3, 4.
TwoElectronOperator = Callable[
    [
        gaussian.GaussianBasis3d,
        gaussian.GaussianBasis3d,
        gaussian.GaussianBasis3d,
        gaussian.GaussianBasis3d,
    ],
    np.ndarray,
]


class _PairIntegralParams(NamedTuple):
    max_degree1: int
    max_degree2: int
    center1: jax.Array
    center2: jax.Array
    cartesian_indices: tuple[jax.Array, ...]
    operator: OneElectronOperator


def _compute_pair_integral(
    params: _PairIntegralParams, exponent1: jax.Array, exponent2: jax.Array
) -> jax.Array:
    """Computes the one-electron integral tensor between two primitive
    Cartesian Gaussians.

    Returns:
        A jax array of shape (n_cart1, n_cart2)
    """
    g1 = gaussian.GaussianBasis3d(params.max_degree1, exponent1, params.center1)
    g2 = gaussian.GaussianBasis3d(params.max_degree2, exponent2, params.center2)

    raw_tensor = params.operator(g1, g2)
    return raw_tensor[params.cartesian_indices]


def one_electron_matrix_jax(
    block1: basis_block.BasisBlock,
    block2: basis_block.BasisBlock,
    operator: OneElectronOperator,
) -> jax.Array:
    """Computes the overlap matrix between two BasisBlocks.

    Returns:
        A jax array of shape (N_basis1, N_basis2) where N_basis1 is the
        number of basis functions in block1 and N_basis2 is the number of basis
        functions in block2.
    """
    # Indices used to broadcast tensors to matrices.
    cartesian_indices = broadcasting.broadcast_indices(
        block1.cartesian_powers, block2.cartesian_powers
    )

    # Generate all pairs of exponents.
    # flat_exps1 and flat_exps2 have shape (n_exps1*n_exps2,)
    flat_exps1, flat_exps2 = broadcasting.flat_product(
        block1.exponents, block2.exponents
    )

    params = _PairIntegralParams(
        max_degree1=block1.max_degree,
        max_degree2=block2.max_degree,
        center1=block1.center,
        center2=block2.center,
        cartesian_indices=cartesian_indices,
        operator=operator,
    )
    integral_fn = functools.partial(_compute_pair_integral, params)

    # Integrals for each pair of exponents.
    # Shape: (n_exps1*n_exps2, n_cart1, n_cart2)
    primitive_integrals_flat = jax.vmap(integral_fn)(flat_exps1, flat_exps2)

    # Reshape to (n_exps1, n_exps2, n_cart1, n_cart2)
    primitive_integrals = primitive_integrals_flat.reshape(
        (
            block1.n_exponents,
            block2.n_exponents,
            block1.n_cart,
            block2.n_cart,
        )
    )

    # Contract over the exponents.
    # Dimensions:
    #   d: n_cart1. Cartesian functions on block 1.
    #   e: n_cart2. Cartesian functions on block 2.
    #   a: n_exp1. Exponents on block 1
    #   b: n_exp2. Exponents on block 2
    # output shape: (n_cart1, n_cart2)
    cartesian_matrix = jnp.einsum(
        "da, eb, abde -> de",
        block1.contraction_matrix,  # shape (n_cart1, n_exp1)
        block2.contraction_matrix,  # shape (n_cart2, n_exp2)
        primitive_integrals,  # shape (n_exp1, n_exp2, n_cart1, n_cart2)
        optimize=True,
    )

    # Transform to the contracted basis representation.
    # shape (n_basis1, n_basis2)
    basis_matrix = (
        block1.basis_transform @ cartesian_matrix @ block2.basis_transform.T
    )

    return basis_matrix


def one_electron_matrix(
    block1: basis_block.BasisBlock,
    block2: basis_block.BasisBlock,
    operator: OneElectronOperator,
) -> np.ndarray:
    jitted_fn = jit(one_electron_matrix_jax, static_argnames="operator")
    return np.array(jitted_fn(block1, block2, operator))


def two_electron_matrix(
    block1: basis_block.BasisBlock,
    block2: basis_block.BasisBlock,
    block3: basis_block.BasisBlock,
    block4: basis_block.BasisBlock,
    operator: TwoElectronOperator,
) -> np.ndarray:
    """Computes the matrix elements for a two-electron operator.

    Returns:
      A numpy array of shape
      (num_basis1, num_basis2, num_basis3, num_basis4)
    """
    d1 = np.max(block1.cartesian_powers)
    d2 = np.max(block2.cartesian_powers)
    d3 = np.max(block3.cartesian_powers)
    d4 = np.max(block4.cartesian_powers)

    # Prepare Cartesian indices for broadcasting.
    ix, iy, iz = [p[:, None, None, None] for p in block1.cartesian_powers.T]
    jx, jy, jz = [p[None, :, None, None] for p in block2.cartesian_powers.T]
    kx, ky, kz = [p[None, None, :, None] for p in block3.cartesian_powers.T]
    lx, ly, lz = [p[None, None, None, :] for p in block4.cartesian_powers.T]

    # Initialize the contracted Cartesian matrix elements
    # Shape: (N_cart1, N_cart2, N_cart3, N_cart4)
    dims = (
        block1.cartesian_powers.shape[0],
        block2.cartesian_powers.shape[0],
        block3.cartesian_powers.shape[0],
        block4.cartesian_powers.shape[0],
    )
    cartesian_matrix = np.zeros(dims, dtype=np.float64)

    # Zip the exponents with the contraction coefficient columns.
    prims1 = zip(block1.exponents, block1.contraction_matrix.T)
    prims2 = zip(block2.exponents, block2.contraction_matrix.T)
    prims3 = zip(block3.exponents, block3.contraction_matrix.T)
    prims4 = zip(block4.exponents, block4.contraction_matrix.T)

    for (a1, c1), (a2, c2), (a3, c3), (a4, c4) in itertools.product(
        prims1, prims2, prims3, prims4
    ):

        # Create primitive Gaussians
        g1 = gaussian.GaussianBasis3d(d1, a1, block1.center)
        g2 = gaussian.GaussianBasis3d(d2, a2, block2.center)
        g3 = gaussian.GaussianBasis3d(d3, a3, block3.center)
        g4 = gaussian.GaussianBasis3d(d4, a4, block4.center)

        # shape (d1+1,)*3 + (d2+1,)*3 + (d3+1,)*3 + (d4+1,)*3
        primitive_tensor = operator(g1, g2, g3, g4)

        # shape (n_cart1, n_cart2, n_cart3, n_cart4)
        primitive_slice = primitive_tensor[
            ix, iy, iz, jx, jy, jz, kx, ky, kz, lx, ly, lz
        ]

        # Broadcast the contraction coefficients.
        c_prod = (
            c1[:, None, None, None]
            * c2[None, :, None, None]
            * c3[None, None, :, None]
            * c4[None, None, None, :]
        )

        cartesian_matrix += c_prod * primitive_slice

    # Transform from Cartesian to the basis coordinates.
    # shape (N_cart1, N_cart2, N_cart3, N_cart4) ->
    #       (N_basis1, N_basis2, N_basis3, N_basis4)
    return np.einsum(
        "pi,qj,rk,sl,ijkl->pqrs",
        block1.basis_transform,  # (N_basis1, N_cart1)
        block2.basis_transform,  # (N_basis2, N_cart2)
        block3.basis_transform,  # (N_basis3, N_cart3)
        block4.basis_transform,  # (N_basis4, N_cart4)
        cartesian_matrix,  # (N_cart1, N_cart2, N_cart3, N_cart4)
        optimize=True,
    )
