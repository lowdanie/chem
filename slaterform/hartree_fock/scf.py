import dataclasses
from typing import Callable, Optional
import functools

import jax
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class


from slaterform.hartree_fock import density
from slaterform.hartree_fock import fock
from slaterform.hartree_fock import one_electron
from slaterform.hartree_fock import roothaan
from slaterform.structure import batched_basis
from slaterform.structure import molecule as molecule_lib
from slaterform.structure import nuclear
from slaterform import types

SolverCallback = Callable[["State"], None]


@register_pytree_node_class
@dataclasses.dataclass
class Options:
    max_iterations: types.IntScalar = 50
    convergence_threshold: types.Scalar = 1e-6
    callback_interval: types.IntScalar = 10
    callback: Optional[SolverCallback] = None

    def __post_init__(self):
        if isinstance(self.callback_interval, (int, float)):
            if self.callback_interval < 1:
                raise ValueError("callback_interval must be >= 1")

        types.promote_dataclass_fields(self)

    def tree_flatten(self):
        children = (
            self.max_iterations,
            self.convergence_threshold,
            self.callback_interval,
        )
        aux_data = (self.callback,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            max_iterations=children[0],
            convergence_threshold=children[1],
            callback_interval=children[2],
            callback=aux_data[0],
        )


@register_pytree_node_class
@dataclasses.dataclass
class Context:
    basis: batched_basis.BatchedBasis

    nuclear_energy: jax.Array

    # Overlap matrix. shape (n_basis, n_basis)
    S: jax.Array

    # Orthogonalizer matrix. shape (n_basis, n_basis)
    X: jax.Array

    # Core Hamiltonian matrix. shape (n_basis, n_basis)
    H_core: jax.Array

    def tree_flatten(self):
        children = (
            self.basis,
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


@register_pytree_node_class
@dataclasses.dataclass
class Result:
    converged: jax.Array
    iterations: jax.Array

    electronic_energy: jax.Array
    nuclear_energy: jax.Array
    total_energy: jax.Array

    # The basis used in the calculation.
    basis: batched_basis.BatchedBasis

    # Fock matrix eigenvalues.
    # shape (n_basis, )
    orbital_energies: jax.Array

    # The molecular orbital coefficients matrix
    # shape (n_basis, n_basis)
    orbitals: jax.Array

    # The closed shell density matrix.
    # shape (n_basis, n_basis)
    density: jax.Array

    def tree_flatten(self):
        children = (
            self.converged,
            self.iterations,
            self.electronic_energy,
            self.nuclear_energy,
            self.total_energy,
            self.basis,
            self.orbital_energies,
            self.orbitals,
            self.density,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def build_initial_state(basis: batched_basis.BatchedBasis) -> State:
    n_basis = basis.n_basis
    S = one_electron.overlap_matrix(basis)
    H_core = one_electron.core_hamiltonian_matrix(basis)
    nuclear_energy = nuclear.repulsion_energy(basis.atoms)

    return State(
        iteration=jnp.array(0, dtype=jnp.int32),
        context=Context(
            basis=basis,
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


def build_result(state: State, options: Options) -> Result:
    return Result(
        converged=state.delta_P <= options.convergence_threshold,
        iterations=state.iteration,
        electronic_energy=state.electronic_energy,
        nuclear_energy=state.context.nuclear_energy,
        total_energy=state.total_energy,
        basis=state.context.basis,
        orbital_energies=state.orbital_energies,
        orbitals=state.C,
        density=state.P,
    )


def scf_step(state: State) -> State:
    # Solve for new orbital coefficients and density.
    # C has shape (n_basis, n_basis)
    orbital_energies, C = roothaan.solve(state.F, state.context.X)
    # shape (n_basis, n_basis)
    P = density.closed_shell_matrix(C, state.context.basis.n_electrons)

    # Compute the Fock matrix and energy for the new density P.
    # shape (n_basis, n_basis)
    G = fock.two_electron_matrix(state.context.basis, P)
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


def _build_basis(
    system: batched_basis.BatchedBasis | molecule_lib.Molecule,
) -> batched_basis.BatchedBasis:
    if isinstance(system, batched_basis.BatchedBasis):
        return system
    elif isinstance(system, molecule_lib.Molecule):
        return batched_basis.BatchedBasis.from_molecule(system)
    else:
        raise TypeError(
            f"Expected input of type BatchedBasis or Molecule, got {type(system)}"
        )


def _should_continue(state: State, options: Options) -> jax.Array:
    return jnp.logical_and(
        state.delta_P > options.convergence_threshold,
        state.iteration < options.max_iterations,
    )


def _maybe_run_callback(state: State, options: Options) -> None:
    if options.callback is None:
        return

    should_run = state.iteration % options.callback_interval == 0

    def run_callback():
        jax.debug.callback(options.callback, state)

    def noop_callback():
        return None

    jax.lax.cond(should_run, run_callback, noop_callback)


def _perform_step(state: State, options: Options) -> State:
    state = scf_step(state)
    _maybe_run_callback(state, options)

    return state


def solve(
    system: batched_basis.BatchedBasis | molecule_lib.Molecule,
    options: Options = Options(),
) -> Result:
    """Performs the self-consistent field (SCF) procedure to compute the
    molecular orbital coefficients and energy.

    Returns:
        A Result object containing the final energy and orbital coefficients.
    """
    basis = _build_basis(system)
    state = build_initial_state(basis)

    cond_fn = functools.partial(_should_continue, options=options)
    step_fn = functools.partial(_perform_step, options=options)
    state = jax.lax.while_loop(cond_fn, step_fn, state)

    return build_result(state, options)
