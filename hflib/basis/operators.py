import dataclasses
import itertools
from typing import Callable

import numpy as np

from hflib.basis import basis_block
from hflib.integrals import gaussian

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


def one_electron_matrix(
    block1: basis_block.BasisBlock,
    block2: basis_block.BasisBlock,
    operator: OneElectronOperator,
) -> np.ndarray:
    """Computes the overlap matrix between two BasisBlocks.

    Returns:
        A numpy array of shape (N_basis1, N_basis2) where N_basis1 is the
        number of basis functions in block1 and N_basis2 is the number of basis
        functions in block2.
    """
    d1 = np.max(block1.cartesian_powers)
    d2 = np.max(block2.cartesian_powers)

    # Prepare Cartesian indices for broadcasting.
    ix, iy, iz = block1.cartesian_powers.T
    ix, iy, iz = ix[:, None], iy[:, None], iz[:, None]
    jx, jy, jz = block2.cartesian_powers.T
    jx, jy, jz = jx[None, :], jy[None, :], jz[None, :]

    # Initialize the contracted Cartesian matrix.
    n_cart1 = block1.cartesian_powers.shape[0]
    n_cart2 = block2.cartesian_powers.shape[0]
    cartesian_matrix = np.zeros((n_cart1, n_cart2), dtype=np.float64)

    # Zip the exponents and coefficient columns of each block.
    prims1 = zip(block1.exponents, block1.contraction_matrix.T)
    prims2 = zip(block2.exponents, block2.contraction_matrix.T)

    # Compute the one-electron integrals for each pair of primitive Cartesian
    # Gaussians.
    for (a1, c1), (a2, c2) in itertools.product(prims1, prims2):
        g1 = gaussian.GaussianBasis3d(
            max_degree=d1, exponent=a1, center=block1.center
        )
        g2 = gaussian.GaussianBasis3d(
            max_degree=d2, exponent=a2, center=block2.center
        )

        # shape (d1+1, d1+1, d1+1, d2+1, d2+1, d2+1)
        primitive_tensor = operator(g1, g2)

        # shape (n_cart1, n_cart2)
        primitive_matrix = primitive_tensor[ix, iy, iz, jx, jy, jz]

        # Accumulate into the contracted Cartesian matrix.
        cartesian_matrix += np.outer(c1, c2) * primitive_matrix

    # Transform to the contracted basis representation.
    # shape (n_basis1, n_basis2)
    basis_matrix = np.dot(
        block1.basis_transform,
        np.dot(cartesian_matrix, block2.basis_transform.T),
    )

    return basis_matrix


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
