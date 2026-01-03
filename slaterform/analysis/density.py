from collections.abc import Sequence

import numpy as np

from slaterform.analysis import grid as grid_lib
from slaterform.basis.basis_block import BasisBlock
from slaterform.structure.real_space import evaluate as evaluate_basis


def evaluate(
    basis_blocks: Sequence[BasisBlock],
    P: np.ndarray,
    grid: grid_lib.RegularGrid,
) -> np.ndarray:
    """Evaluate the electron density at given grid points.

    Args:
        mol_basis: The molecular basis set.
        P: The density matrix of shape (n_basis, n_basis).
        grid: The regular grid on which to evaluate the density.

    Returns:
        The electron density evaluated at each grid point,
        shape grid.dims
    """
    points = grid_lib.generate_points(grid)

    # Evaluate the basis functions phi at the specified points.
    # shape (..., n_basis)
    phi = evaluate_basis(basis_blocks, points)

    # Compute the density using the formula:
    # rho(r) = phi(r) @ P @ phi(r).T

    # phi: shape (..., n_basis)
    # P: shape (n_basis, n_basis)
    # phi_P: shape (..., n_basis)
    phi_P = np.matmul(phi, P)

    # phi_P: shape (..., n_basis)
    # phi: shape (..., n_basis)
    # rho: shape (...)
    rho = np.sum(phi_P * phi, axis=-1)

    return rho
