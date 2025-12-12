import dataclasses
from typing import Callable, Optional

import numpy as np

from slaterform.hartree_fock import density
from slaterform.hartree_fock import fock
from slaterform.hartree_fock import one_electron
from slaterform.hartree_fock import roothaan
from slaterform.structure import molecular_basis

SolverCallback = Callable[["State"], None]


@dataclasses.dataclass
class Options:
    max_iterations: int = 50
    convergence_threshold: float = 1e-6
    callback: Optional[SolverCallback] = None


@dataclasses.dataclass
class Context:
    # Overlap matrix. shape (n_basis, n_basis)
    S: np.ndarray

    # Orthogonalizer matrix. shape (n_basis, n_ind)
    X: np.ndarray

    # Core Hamiltonian matrix. shape (n_basis, n_basis)
    H_core: np.ndarray


@dataclasses.dataclass
class State:
    iteration: int
    context: Context

    # Molecular orbital coefficients matrix. shape (n_basis, n_ind)
    C: np.ndarray

    # Closed shell density matrix. shape (n_basis, n_basis)
    P: np.ndarray

    # Fock matrix. shape (n_basis, n_basis)
    F: np.ndarray

    # The electronic energy.
    electronic_energy: np.float64

    # Fock matrix eigenvalues. shape (n_basis, )
    orbital_energies: np.ndarray

    # Change in density matrix. ||P_new - P_old||_2
    delta_P: np.float64


@dataclasses.dataclass
class Result:
    converged: bool
    iterations: int

    electronic_energy: np.float64

    # Fock matrix eigenvalues.
    # shape (n_basis, )
    orbital_energies: np.ndarray

    # The molecular orbital coefficients matrix
    # shape (n_basis, n_basis)
    orbitals: np.ndarray

    # The closed shell density matrix.
    # shape (n_basis, n_basis)
    density: np.ndarray


def _build_initial_state(mol_basis: molecular_basis.MolecularBasis) -> State:
    n_basis = mol_basis.n_basis
    S = one_electron.overlap_matrix(mol_basis)
    H_core = one_electron.core_hamiltonian_matrix(mol_basis)

    return State(
        iteration=0,
        context=Context(
            S=S,
            X=roothaan.orthogonalize_basis(S),
            H_core=H_core,
        ),
        C=np.zeros((n_basis, n_basis), dtype=np.float64),
        P=np.zeros((n_basis, n_basis), dtype=np.float64),
        F=H_core,
        electronic_energy=np.float64(0.0),
        orbital_energies=np.zeros(n_basis, dtype=np.float64),
        delta_P=np.float64(np.inf),
    )


def _build_result(state: State, converged: bool) -> Result:
    return Result(
        converged=converged,
        iterations=state.iteration,
        electronic_energy=state.electronic_energy,
        orbital_energies=state.orbital_energies,
        orbitals=state.C,
        density=state.P,
    )


def _scf_step(
    state: State,
    mol_basis: molecular_basis.MolecularBasis,
) -> State:
    # Solve for new orbital coefficients and density.
    # C has shape (n_basis, n_ind)
    orbital_energies, C = roothaan.solve(state.F, state.context.X)
    P = density.closed_shell_matrix(C, mol_basis.n_electrons)

    # Compute the Fock matrix and energy for the new density P.
    G = fock.two_electron_matrix(mol_basis, P)  # shape (n_basis, n_basis)
    F = state.context.H_core + G  # shape (n_basis, n_basis)
    electronic_energy = fock.electronic_energy(state.context.H_core, F, P)

    return State(
        iteration=state.iteration + 1,
        context=state.context,
        C=C,
        P=P,
        F=F,
        electronic_energy=electronic_energy,
        orbital_energies=orbital_energies,
        delta_P=np.linalg.norm(P - state.P),
    )


def solve(
    mol_basis: molecular_basis.MolecularBasis,
    options: Options = Options(),
) -> Result:
    """Performs the self-consistent field (SCF) procedure to compute the
    molecular orbital coefficients and energy.

    Returns:
        A Result object containing the final energy and orbital coefficients.
    """
    state = _build_initial_state(mol_basis)
    converged = False

    for _ in range(options.max_iterations):
        state = _scf_step(state, mol_basis)

        if options.callback is not None:
            options.callback(state)

        # Check for convergence.
        if state.delta_P < options.convergence_threshold:
            converged = True
            break

    return _build_result(state, converged)
