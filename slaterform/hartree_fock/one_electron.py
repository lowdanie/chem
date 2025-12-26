from collections.abc import Sequence
from typing import Callable, NamedTuple
import functools

import jax
from jax import jit
from jax import numpy as jnp
import numpy as np


from slaterform.integrals import gaussian
from slaterform.integrals import overlap
from slaterform.integrals import kinetic
from slaterform.integrals import coulomb
from slaterform.structure import atom
from slaterform.structure import molecular_basis
from slaterform.structure import batched_molecular_basis as bmb
from slaterform.basis import basis_block
from slaterform.basis import operators
from slaterform.jax_utils import batching
from slaterform.jax_utils import scatter

_BlockOperator = Callable[
    [basis_block.BasisBlock, basis_block.BasisBlock], jax.Array
]


class _GroupParams(NamedTuple):
    # The stacked basis blocks for the group of batches.
    # Length: 2
    stacks: tuple[basis_block.BasisBlock, ...]

    # The starting indices of each basis block in the stack with respect to
    # the full basis set.
    # Length: 2
    stack_starts: tuple[jax.Array, ...]

    # A block operator that can operate on basis blocks with a batch dimension.
    batch_operator: _BlockOperator


class _BatchData(NamedTuple):
    tuple_indices: jax.Array  # Shape (batch_size, 2)
    mask: jax.Array  # Shape (batch_size,)


def _batch_step(
    params: _GroupParams, matrix: jax.Array, batch_data: _BatchData
) -> tuple[jax.Array, None]:
    # Indices of each pair of blocks in the batch.
    # shape: (batch_size,)
    i_idx = batch_data.tuple_indices[:, 0]
    j_idx = batch_data.tuple_indices[:, 1]

    # Gather the basis blocks for each pair in the batch.
    # Each has shape (batch_size, ...)
    i_block = jax.tree.map(lambda x: x[i_idx], params.stacks[0])
    j_block = jax.tree.map(lambda x: x[j_idx], params.stacks[1])

    # Compute the one-electron integrals for the batch.
    # shape: (batch_size, n_i_basis, n_j_basis)
    integral_matrix = params.batch_operator(i_block, j_block)

    new_matrix = scatter.add_tiles_2d(
        matrix=matrix,
        tiles=integral_matrix,
        row_starts=params.stack_starts[0][i_idx],
        col_starts=params.stack_starts[1][j_idx],
        mask=batch_data.mask,
    )

    # Also add the transpose for the symmetric entry if i!=j
    new_matrix = scatter.add_tiles_2d(
        matrix=new_matrix,
        tiles=integral_matrix.transpose((0, 2, 1)),
        row_starts=params.stack_starts[1][j_idx],
        col_starts=params.stack_starts[0][i_idx],
        mask=batch_data.mask * (i_idx != j_idx),
    )

    return new_matrix, None


def _process_batched_tuples(
    matrix: jax.Array,
    batched_tuples: batching.BatchedTreeTuples,
    block_starts: jax.Array,
    batch_operator: _BlockOperator,
) -> jax.Array:
    """Compute contributions to the one-electron matrix from batched tuples."""
    stack_starts = tuple(
        block_starts[idx] for idx in batched_tuples.global_tree_indices
    )
    params = _GroupParams(
        stacks=batched_tuples.stacks,
        stack_starts=stack_starts,
        batch_operator=batch_operator,
    )
    batch_data = _BatchData(
        tuple_indices=batched_tuples.tuple_indices,
        mask=batched_tuples.padding_mask,
    )
    scan_fn = functools.partial(_batch_step, params)
    new_matrix, _ = jax.lax.scan(scan_fn, matrix, batch_data)

    return new_matrix


def _one_electron_matrix_jax(
    basis: bmb.BatchedMolecularBasis,
    operator: operators.OneElectronOperator,
) -> jax.Array:
    n_basis = basis.basis.n_basis
    matrix = jnp.zeros((n_basis, n_basis), dtype=jnp.float64)
    batch_operator = jax.vmap(
        lambda b1, b2: operators.one_electron_matrix_jax(b1, b2, operator)
    )

    for batched_tuples in basis.batches_1e:
        matrix = _process_batched_tuples(
            matrix,
            batched_tuples,
            jnp.asarray(basis.block_starts),
            batch_operator,
        )

    return matrix


# def one_electron_matrix(
#     mol_basis: molecular_basis.MolecularBasis,
#     operator: operators.OneElectronOperator,
# ) -> np.ndarray:
#     """Computes the one-electron matrix for a given molecular basis.

#     Returns:
#         A numpy array of shape (N, N) where N=mol_basis.n_basis
#     """
#     output = np.empty((mol_basis.n_basis, mol_basis.n_basis), dtype=np.float64)

#     for i, j in itertools.combinations_with_replacement(
#         range(len(mol_basis.basis_blocks)), 2
#     ):
#         block1, slice1 = mol_basis.basis_blocks[i], mol_basis.block_slices[i]
#         block2, slice2 = mol_basis.basis_blocks[j], mol_basis.block_slices[j]

#         block_matrix = operators.one_electron_matrix(block1, block2, operator)

#         output[slice1, slice2] = block_matrix

#         if i != j:
#             output[slice2, slice1] = block_matrix.T

#     return output


def overlap_matrix_jax(
    batched_mol_basis: bmb.BatchedMolecularBasis,
) -> jax.Array:
    """Computes the overlap matrix S

    Returns:
        A numpy array of shape (N, N) where N=mol_basis.n_basis
    """
    S = _one_electron_matrix_jax(
        batched_mol_basis,
        overlap.overlap_3d,
    )

    return S


def overlap_matrix(
    basis: bmb.BatchedMolecularBasis,
) -> np.ndarray:
    return np.array(jit(overlap_matrix_jax)(basis))


# def nuclear_attraction_matrix(
#     mol_basis: molecular_basis.MolecularBasis,
# ) -> np.ndarray:
#     """Computes the nuclear attraction matrix V

#     Returns:
#         A numpy array of shape (N, N) where N=mol_basis.n_basis
#     """
#     V = np.zeros((mol_basis.n_basis, mol_basis.n_basis), dtype=np.float64)
#     for atom in mol_basis.atoms:
#         V -= atom.number * one_electron_matrix(
#             mol_basis,
#             lambda g1, g2: coulomb.one_electron(g1, g2, atom.position),
#         )

#     return V


def _nuclear_operator(
    atomic_positions: jax.Array,
    atomic_charges: jax.Array,
    g1: gaussian.GaussianBasis3d,
    g2: gaussian.GaussianBasis3d,
) -> jax.Array:
    tensors = jax.vmap(
        lambda R, Z: -Z * coulomb.one_electron(g1, g2, R),
        in_axes=(0, 0),
    )(atomic_positions, atomic_charges)

    return jnp.sum(tensors, axis=0)


def nuclear_attraction_matrix_jax(
    basis: bmb.BatchedMolecularBasis,
) -> jax.Array:
    """Computes the nuclear attraction matrix V

    Returns:
        A numpy array of shape (N, N) where N=mol_basis.n_basis
    """
    positions = jnp.asarray([atom.position for atom in basis.basis.atoms])
    charges = jnp.asarray([atom.number for atom in basis.basis.atoms])
    nuclear_op = functools.partial(_nuclear_operator, positions, charges)

    V = _one_electron_matrix_jax(basis, nuclear_op)

    return V


def nuclear_attraction_matrix(
    basis: bmb.BatchedMolecularBasis,
) -> np.ndarray:
    return np.array(jit(nuclear_attraction_matrix_jax)(basis))


def kinetic_matrix_jax(
    basis: bmb.BatchedMolecularBasis,
) -> jax.Array:
    """Computes the kinetic energy matrix T

    Returns:
        A numpy array of shape (N, N) where N=mol_basis.n_basis
    """
    return -0.5 * _one_electron_matrix_jax(basis, kinetic.kinetic_3d)


def kinetic_matrix(
    basis: bmb.BatchedMolecularBasis,
) -> np.ndarray:
    """Computes the kinetic energy matrix T

    Returns:
        A numpy array of shape (N, N) where N=mol_basis.n_basis
    """
    return np.array(jit(kinetic_matrix_jax)(basis))


def core_hamiltonian_matrix_jax(
    basis: bmb.BatchedMolecularBasis,
) -> jax.Array:
    """Computes the core Hamiltonian matrix H = T + V

    Returns:
        A numpy array of shape (N, N) where N=mol_basis.n_basis
    """
    T = kinetic_matrix_jax(basis)
    V = nuclear_attraction_matrix_jax(basis)

    return T + V


def core_hamiltonian_matrix(
    basis: bmb.BatchedMolecularBasis,
) -> np.ndarray:
    """Computes the core Hamiltonian matrix H = T + V

    Returns:
        A numpy array of shape (N, N) where N=mol_basis.n_basis
    """
    return np.array(jit(core_hamiltonian_matrix_jax)(basis))
