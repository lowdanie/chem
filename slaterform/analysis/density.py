import numpy as np
import numpy.typing as npt

from slaterform.analysis import grid as grid_lib
from slaterform.structure import molecular_basis
from slaterform.structure import real_space


def evaluate_density(
    mol_basis: molecular_basis.MolecularBasis,
    P: npt.NDArray[np.float64],
    grid: grid_lib.RegularGrid,
) -> npt.NDArray[np.float64]:
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
    phi = real_space.evaluate(mol_basis, points)

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
