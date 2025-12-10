import dataclasses

import numpy as np

from hartree_fock import density
from hartree_fock import fock
from hartree_fock import one_electron
from hartree_fock import roothaan
from structure import molecular_basis


@dataclasses.dataclass
class SCFResult:
    electronic_energy: float

    # The molecular orbital coefficients matrix
    # shape (n_basis, n_basis)
    orbitals: np.ndarray


def scf(
    mol_basis: molecular_basis.MolecularBasis,
    max_iterations: int = 100,
    convergence_threshold: float = 1e-6,
) -> SCFResult:
    """Performs the self-consistent field (SCF) procedure to compute the
    molecular orbital coefficients and energy.

    Returns:
        An SCFResult containing the final energy and orbital coefficients.
    """
    n_basis = mol_basis.n_basis

    # Compute the overlap matrix and its orthogonalizer.
    S = one_electron.overlap_matrix(mol_basis)
    X = roothaan.orthogonalize_basis(S)

    H_core = one_electron.core_hamiltonian_matrix(mol_basis)

    # Initialize the density matrix.
    P = np.zeros((n_basis, n_basis), dtype=np.float64)

    # Initialize variables.
    C = np.zeros((n_basis, n_basis), dtype=np.float64)
    F = np.zeros((n_basis, n_basis), dtype=np.float64)
    energy = 0.0

    for iteration in range(max_iterations):
        # Compute the Fock matrix.s
        G = fock.fock_two_electron_matrix(mol_basis, P)
        F = H_core + G

        # Solve for new orbital coefficients.
        _, C = roothaan.solve(F, X)

        # Build new density matrix.
        P_new = density.closed_shell_matrix(C, mol_basis.n_electrons)

        delta_P = np.linalg.norm(P_new - P)
        P = P_new
        energy = fock.electronic_energy(F, P)

        print(
            f"Iteration {iteration}: Electronic Energy = {energy:.10f} "
            f"Delta P = {delta_P:.10e}"
        )

        # Check for convergence.
        if delta_P < convergence_threshold:
            break

    return SCFResult(electronic_energy=energy, orbitals=C)
