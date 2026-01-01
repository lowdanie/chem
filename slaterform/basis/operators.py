from typing import Callable, NamedTuple
import functools

import jax
import jax.numpy as jnp

from slaterform.basis.basis_block import BasisBlock
from slaterform.jax_utils.broadcasting import broadcast_indices, flat_product
from slaterform.integrals.gaussian import GaussianBasis3d

# A one-electron operator between two BasisBlocks.
# The returned array has shape (d1+1, d1+1, d1+1, d2+1, d2+1, d2+1)
# where d1 = basis1.max_degree and d2 = basis2.max_degree.
OneElectronOperator = Callable[[GaussianBasis3d, GaussianBasis3d], jax.Array]

# A two-electron operator between four BasisBlocks.
# The returned array has shape
# (d1+1, d1+1, d1+1,
#  d2+1, d2+1, d2+1,
#  d3+1, d3+1, d3+1,
#  d4+1, d4+1, d4+1)
# where di = the max_degree in basisi.cartesian_powers for i = 1, 2, 3, 4.
TwoElectronOperator = Callable[
    [
        GaussianBasis3d,
        GaussianBasis3d,
        GaussianBasis3d,
        GaussianBasis3d,
    ],
    jax.Array,
]


class _PrimitiveIntegralParams(NamedTuple):
    max_degrees: tuple[int, ...]  # length n_primitives
    centers: tuple[jax.Array, ...]  # length n_primitives
    cartesian_indices: tuple[jax.Array, ...]  # length 3 * n_primitives
    operator: OneElectronOperator | TwoElectronOperator


def _compute_primitive_integral(
    params: _PrimitiveIntegralParams, *exponents: jax.Array
) -> jax.Array:
    """Computes the one-electron integral tensor between two primitive
    Cartesian Gaussians.

    Args:
        params: The static parameters for n_primitives primitive Gaussians.
        exponents: The exponents of the primitive Gaussians. Length n_primitives.
    Returns:
        A jax array of shape (n_cart1, n_cart2)
    """
    gaussian_args = [
        GaussianBasis3d(params.max_degrees[i], exponents[i], params.centers[i])
        for i in range(len(exponents))
    ]

    raw_tensor = params.operator(*gaussian_args)
    return raw_tensor[params.cartesian_indices]


def one_electron_matrix(
    block1: BasisBlock,
    block2: BasisBlock,
    operator: OneElectronOperator,
) -> jax.Array:
    """Computes the overlap matrix between two BasisBlocks.

    Returns:
        A jax array of shape (N_basis1, N_basis2) where N_basis1 is the
        number of basis functions in block1 and N_basis2 is the number of basis
        functions in block2.
    """
    # Indices used to broadcast tensors to matrices.
    cartesian_indices = broadcast_indices(
        block1.cartesian_powers, block2.cartesian_powers
    )

    # Generate all pairs of exponents.
    # flat_exps1 and flat_exps2 have shape (n_exps1*n_exps2,)
    exp_flat_product = flat_product(
        jnp.asarray(block1.exponents), jnp.asarray(block2.exponents)
    )

    params = _PrimitiveIntegralParams(
        max_degrees=(block1.max_degree, block2.max_degree),
        centers=(jnp.asarray(block1.center), jnp.asarray(block2.center)),
        cartesian_indices=cartesian_indices,
        operator=operator,
    )
    integral_fn = functools.partial(_compute_primitive_integral, params)

    # Integrals for each pair of exponents.
    # Shape: (n_exps1*n_exps2, n_cart1, n_cart2)
    primitive_integrals_flat = jax.vmap(integral_fn)(*exp_flat_product)

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


def two_electron_matrix(
    block1: BasisBlock,
    block2: BasisBlock,
    block3: BasisBlock,
    block4: BasisBlock,
    operator: TwoElectronOperator,
) -> jax.Array:
    """Computes the matrix elements for a two-electron operator.

    Returns:
      A jax array of shape
      (num_basis1, num_basis2, num_basis3, num_basis4)
    """
    blocks = (block1, block2, block3, block4)

    # Indices used to broadcast tensors to matrices.
    cartesian_indices = broadcast_indices(
        *tuple(b.cartesian_powers for b in blocks)
    )

    # Generate all pairs of exponents.
    # flat_exps1 and flat_exps2 have shape (n_exps1*n_exps2,)
    exp_flat_product = flat_product(
        *tuple(jnp.asarray(b.exponents) for b in blocks)
    )

    params = _PrimitiveIntegralParams(
        max_degrees=tuple(b.max_degree for b in blocks),
        centers=tuple(jnp.asarray(b.center) for b in blocks),
        cartesian_indices=cartesian_indices,
        operator=operator,
    )
    integral_fn = functools.partial(_compute_primitive_integral, params)

    # Integrals for each quartet of exponents.
    # Shape: (n_exps1*...*n_exps4, n_cart1, ..., n_cart4)
    primitive_integrals_flat = jax.vmap(integral_fn)(*exp_flat_product)

    # Reshape to (n_exps1, ..., n_exps4, n_cart1, ..., n_cart4)
    primitive_integrals = primitive_integrals_flat.reshape(
        tuple(b.n_exponents for b in blocks) + tuple(b.n_cart for b in blocks)
    )

    # Contract over the exponents.
    # Dimensions:
    #   i: n_cart1. Cartesian functions on block 1.
    #   j: n_cart2. Cartesian functions on block 2.
    #   k: n_cart3. Cartesian functions on block 3.
    #   l: n_cart4. Cartesian functions on block 4.
    #   a: n_exp1. Exponents on block 1
    #   b: n_exp2. Exponents on block 2
    #   c: n_exp3. Exponents on block 3
    #   d: n_exp4. Exponents on block 4
    # output shape: (n_cart1, ..., n_cart4)
    cartesian_matrix = jnp.einsum(
        "ia, jb, kc, ld, abcdijkl -> ijkl",
        block1.contraction_matrix,  # shape (n_cart1, n_exp1)
        block2.contraction_matrix,  # shape (n_cart2, n_exp2)
        block3.contraction_matrix,  # shape (n_cart3, n_exp3)
        block4.contraction_matrix,  # shape (n_cart4, n_exp4)
        primitive_integrals,  # shape (n_exp1, ... , n_exp4, n_cart1, .., n_cart4)
        optimize=True,
    )

    # Transform to the contracted basis representation.
    # shape (n_basis1, n_basis2)
    return jnp.einsum(
        "pi,qj,rk,sl,ijkl->pqrs",
        block1.basis_transform,  # (n_basis1, n_cart1)
        block2.basis_transform,  # (n_basis2, n_cart2)
        block3.basis_transform,  # (n_basis3, n_cart3)
        block4.basis_transform,  # (n_basis4, n_cart4)
        cartesian_matrix,  # (n_cart1, n_cart2, n_cart3, n_cart4)
        optimize=True,
    )
