import dataclasses
from typing import Callable, Optional
import time

import jax
from jax import jit
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class


from slaterform.hartree_fock import density
from slaterform.hartree_fock import fock
from slaterform.hartree_fock import one_electron
from slaterform.hartree_fock import roothaan
from slaterform.structure import batched_molecular_basis as bmb
from slaterform.structure import nuclear
from slaterform import types

SolverCallback = Callable[["State"], None]


@dataclasses.dataclass
class Options:
    max_iterations: int = 50
    convergence_threshold: float = 1e-6
    callback: Optional[SolverCallback] = None


@register_pytree_node_class
@dataclasses.dataclass
class Context:
    nuclear_energy: jax.Array

    # Overlap matrix. shape (n_basis, n_basis)
    S: jax.Array

    # Orthogonalizer matrix. shape (n_basis, n_basis)
    X: jax.Array

    # Core Hamiltonian matrix. shape (n_basis, n_basis)
    H_core: jax.Array

    def tree_flatten(self):
        children = (
            self.nuclear_energy,
            self.S,
            self.X,
            self.H_core,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@register_pytree_node_class
@dataclasses.dataclass
class State:
    iteration: jax.Array
    context: Context

    # Molecular orbital coefficients matrix. shape (n_basis, n_basis)
    C: jax.Array

    # Closed shell density matrix. shape (n_basis, n_basis)
    P: jax.Array

    # Fock matrix. shape (n_basis, n_basis)
    F: jax.Array

    electronic_energy: jax.Array
    total_energy: jax.Array

    # Fock matrix eigenvalues. shape (n_basis, )
    orbital_energies: jax.Array

    # Change in density matrix. ||P_new - P_old||_2
    delta_P: jax.Array

    def tree_flatten(self):
        children = (
            self.iteration,
            self.context,
            self.C,
            self.P,
            self.F,
            self.electronic_energy,
            self.total_energy,
            self.orbital_energies,
            self.delta_P,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@dataclasses.dataclass
class Result:
    converged: bool
    iterations: int

    electronic_energy: jax.Array
    nuclear_energy: jax.Array
    total_energy: jax.Array

    # Fock matrix eigenvalues.
    # shape (n_basis, )
    orbital_energies: jax.Array

    # The molecular orbital coefficients matrix
    # shape (n_basis, n_basis)
    orbitals: jax.Array

    # The closed shell density matrix.
    # shape (n_basis, n_basis)
    density: jax.Array


def build_initial_state(basis: bmb.BatchedMolecularBasis) -> State:
    n_basis = basis.basis.n_basis
    S = one_electron.overlap_matrix(basis)
    H_core = one_electron.core_hamiltonian_matrix(basis)
    nuclear_energy = nuclear.repulsion_energy(basis.basis.molecule)

    return State(
        iteration=jnp.array(0, dtype=jnp.int32),
        context=Context(
            nuclear_energy=nuclear_energy,
            S=S,
            X=roothaan.orthogonalize_basis(S),
            H_core=H_core,
        ),
        C=jnp.zeros((n_basis, n_basis), dtype=jnp.float64),
        P=jnp.zeros((n_basis, n_basis), dtype=jnp.float64),
        F=H_core,
        electronic_energy=jnp.asarray(0.0, dtype=jnp.float64),
        total_energy=nuclear_energy,
        orbital_energies=jnp.zeros(n_basis, dtype=jnp.float64),
        delta_P=jnp.array(jnp.inf, dtype=jnp.float64),
    )


def build_result(state: State, converged: bool) -> Result:
    return Result(
        converged=converged,
        iterations=state.iteration.item(),
        electronic_energy=state.electronic_energy,
        nuclear_energy=state.context.nuclear_energy,
        total_energy=state.total_energy,
        orbital_energies=state.orbital_energies,
        orbitals=state.C,
        density=state.P,
    )


def scf_step(
    state: State,
    basis: bmb.BatchedMolecularBasis,
) -> State:
    # Solve for new orbital coefficients and density.
    # C has shape (n_basis, n_basis)
    orbital_energies, C = roothaan.solve(state.F, state.context.X)
    # shape (n_basis, n_basis)
    P = density.closed_shell_matrix(C, basis.basis.n_electrons)

    # Compute the Fock matrix and energy for the new density P.
    G = fock.two_electron_matrix(basis, P)  # shape (n_basis, n_basis)
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
        delta_P=jnp.linalg.norm(P - state.P),
    )


def solve(
    basis: bmb.BatchedMolecularBasis,
    options: Options = Options(),
) -> Result:
    """Performs the self-consistent field (SCF) procedure to compute the
    molecular orbital coefficients and energy.

    Returns:
        A Result object containing the final energy and orbital coefficients.
    """
    state = build_initial_state(basis)
    converged = False

    step_fn = jit(scf_step)

    for _ in range(options.max_iterations):
        state = step_fn(state, basis)

        if options.callback is not None:
            options.callback(state)

        # Check for convergence.
        if state.delta_P.item() < options.convergence_threshold:
            converged = True
            break

    return build_result(state, converged)
