import dataclasses
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt

from slaterform.hartree_fock import density
from slaterform.hartree_fock import fock
from slaterform.hartree_fock import one_electron
from slaterform.hartree_fock import roothaan
from slaterform.structure import molecular_basis
from slaterform.structure import nuclear

SolverCallback = Callable[["State"], None]


@dataclasses.dataclass
class Options:
    max_iterations: int = 50
    convergence_threshold: float = 1e-6
    callback: Optional[SolverCallback] = None


@dataclasses.dataclass
class Context:
    nuclear_energy: float

    # Overlap matrix. shape (n_basis, n_basis)
    S: npt.NDArray[np.float64]

    # Orthogonalizer matrix. shape (n_basis, n_ind)
    X: npt.NDArray[np.float64]

    # Core Hamiltonian matrix. shape (n_basis, n_basis)
    H_core: npt.NDArray[np.float64]


@dataclasses.dataclass
class State:
    iteration: int
    context: Context

    # Molecular orbital coefficients matrix. shape (n_basis, n_ind)
    C: npt.NDArray[np.float64]

    # Closed shell density matrix. shape (n_basis, n_basis)
    P: npt.NDArray[np.float64]

    # Fock matrix. shape (n_basis, n_basis)
    F: npt.NDArray[np.float64]

    electronic_energy: float
    total_energy: float

    # Fock matrix eigenvalues. shape (n_basis, )
    orbital_energies: npt.NDArray[np.float64]

    # Change in density matrix. ||P_new - P_old||_2
    delta_P: float


@dataclasses.dataclass
class Result:
    converged: bool
    iterations: int

    electronic_energy: float
    nuclear_energy: float
    total_energy: float

    # Fock matrix eigenvalues.
    # shape (n_basis, )
    orbital_energies: npt.NDArray[np.float64]

    # The molecular orbital coefficients matrix
    # shape (n_basis, n_basis)
    orbitals: npt.NDArray[np.float64]

    # The closed shell density matrix.
    # shape (n_basis, n_basis)
    density: npt.NDArray[np.float64]


def _build_initial_state(mol_basis: molecular_basis.MolecularBasis) -> State:
    n_basis = mol_basis.n_basis
    S = one_electron.overlap_matrix(mol_basis)
    H_core = one_electron.core_hamiltonian_matrix(mol_basis)
    nuclear_energy = nuclear.repulsion_energy(mol_basis.molecule)

    return State(
        iteration=0,
        context=Context(
            nuclear_energy=nuclear_energy,
            S=S,
            X=roothaan.orthogonalize_basis(S),
            H_core=H_core,
        ),
        C=np.zeros((n_basis, n_basis), dtype=np.float64),
        P=np.zeros((n_basis, n_basis), dtype=np.float64),
        F=H_core,
        electronic_energy=0.0,
        total_energy=nuclear_energy,
        orbital_energies=np.zeros(n_basis, dtype=np.float64),
        delta_P=np.inf,
    )


def _build_result(state: State, converged: bool) -> Result:
    return Result(
        converged=converged,
        iterations=state.iteration,
        electronic_energy=state.electronic_energy,
        nuclear_energy=state.context.nuclear_energy,
        total_energy=state.total_energy,
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
        total_energy=electronic_energy + state.context.nuclear_energy,
        orbital_energies=orbital_energies,
        delta_P=float(np.linalg.norm(P - state.P)),
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
