import dataclasses

import numpy as np

from chem.hartree_fock import density
from chem.hartree_fock import fock
from chem.hartree_fock import one_electron
from chem.hartree_fock import roothaan
from chem.structure import molecular_basis


@dataclasses.dataclass
class SCFResult:
    electronic_energy: np.float64

    # The molecular orbital coefficients matrix
    # shape (n_basis, n_basis)
    orbitals: np.ndarray


def solve(
    mol_basis: molecular_basis.MolecularBasis,
    max_iterations: int = 50,
    convergence_threshold: float = 1e-6,
) -> SCFResult:
    """Performs the self-consistent field (SCF) procedure to compute the
    molecular orbital coefficients and energy.

    Returns:
        An SCFResult containing the final energy and orbital coefficients.
    """
    n_basis = mol_basis.n_basis

    # Compute the overlap matrix and its orthogonalizer.
    S = one_electron.overlap_matrix(mol_basis)  # shape (n_basis, n_basis)
    X = roothaan.orthogonalize_basis(S)  # shape (n_basis, n_ind)

    # shape (n_basis, n_basis)
    H_core = one_electron.core_hamiltonian_matrix(mol_basis)

    # Initialize the density matrix.
    P = np.zeros((n_basis, n_basis), dtype=np.float64)

    # Initialize variables.
    C = np.zeros((n_basis, n_basis), dtype=np.float64)
    F = np.zeros((n_basis, n_basis), dtype=np.float64)
    electronic_energy = np.float64(0.0)

    for iteration in range(max_iterations):
        # Compute the Fock matrix and energy for the current density P.
        G = fock.two_electron_matrix(mol_basis, P)  # shape (n_basis, n_basis)
        F = H_core + G  # shape (n_basis, n_basis)
        electronic_energy = fock.electronic_energy(H_core, F, P)

        # Solve for new orbital coefficients and density.
        # C has shape (n_basis, n_ind)
        _, C = roothaan.solve(F, X)
        P_new = density.closed_shell_matrix(C, mol_basis.n_electrons)

        delta_P = np.linalg.norm(P_new - P)
        P = P_new

        print(
            f"Iteration {iteration}: Electronic Energy = {electronic_energy:.10f} "
            f"Delta P = {delta_P:.10e}"
        )

        # Check for convergence.
        if delta_P < convergence_threshold:
            break

    return SCFResult(electronic_energy=electronic_energy, orbitals=C)
