from typing import Callable, NamedTuple
import functools

import jax
from jax import jit
from jax import numpy as jnp
import numpy as np

from slaterform.basis import operators
from slaterform.basis import basis_block
from slaterform.jax_utils import batching
from slaterform.jax_utils import gather
from slaterform.jax_utils import scatter
from slaterform.integrals import coulomb
from slaterform.symmetry import quartet as quartet_lib
from slaterform.structure import molecular_basis
from slaterform.structure import batched_molecular_basis as bmb

_BlockOperator = Callable[
    [
        basis_block.BasisBlock,
        basis_block.BasisBlock,
        basis_block.BasisBlock,
        basis_block.BasisBlock,
    ],
    jax.Array,
]


class _GroupParams(NamedTuple):
    # The density matrix.
    # Shape: (n_basis, n_basis)
    P: jax.Array

    # The stacked basis blocks for the group of batches.
    # Length: 4
    stacks: tuple[basis_block.BasisBlock, ...]

    # The starting indices of each basis block in the stack with respect to
    # the full basis set.
    # Length: 4
    stack_starts: tuple[jax.Array, ...]

    # A block operator that can operate on basis blocks with a batch dimension.
    batch_operator: _BlockOperator


class _BatchData(NamedTuple):
    tuple_indices: jax.Array  # Shape (batch_size, 4)
    mask: jax.Array  # Shape (batch_size,)


def _batch_step(
    params: _GroupParams, G: jax.Array, batch_data: _BatchData
) -> tuple[jax.Array, None]:
    # Indices of the blocks in each quartet in the batch.
    # Shape: (batch_size, 4)
    idx = batch_data.tuple_indices

    # Gather the basis blocks for each quartet in the batch.
    # Length 4. Each is a stacked BasisBlock with shape (batch_size, ...)
    blocks = tuple(
        jax.tree.map(lambda x: x[idx[:, k]], params.stacks[k]) for k in range(4)
    )

    # Get the starting indices of each block in the full basis set.
    # Length: 4. Each has shape (batch_size,)
    block_starts = quartet_lib.BatchedQuartet(
        params.stack_starts[k][idx[:, k]] for k in range(4)
    )

    # Compute the two-electron integrals for the batch.
    # shape: (batch_size, n_i_basis, n_j_basis, n_k_basis, n_l_basis)
    integrals = params.batch_operator(*blocks)

    # Scale the mask by the inverse of the stabilizer norm to account
    # for double counting.
    # Shape: (batch_size, 1, 1, 1, 1)
    stabilizer_norm = quartet_lib.compute_stabilizer_norm(block_starts)
    scaled_mask = batch_data.mask / stabilizer_norm.astype(integrals.dtype)
    scaled_mask = scaled_mask[:, None, None, None, None]

    for sigma in quartet_lib.get_symmetries():
        # Permute integrals and apply the scaled mask.
        # Transpose dims (0, 1+s0, 1+s1...)
        batch_sigma = (0,) + tuple(s + 1 for s in sigma)
        sigma_integrals = jnp.transpose(integrals, batch_sigma) * scaled_mask

        # Permute the indices
        starts_i, starts_j, starts_k, starts_l = quartet_lib.apply_permutation(
            sigma, block_starts
        )

        # Get the shapes of the permuted basis blocks
        _, _, n_j, n_k, n_l = sigma_integrals.shape

        # Coulomb update: G_ij += (ij|kl) * P_lk
        P_lk = gather.extract_tiles_2d(
            matrix=params.P,
            row_starts=starts_l,
            col_starts=starts_k,
            n_rows=n_l,
            n_cols=n_k,
        )
        J_ij = jnp.einsum("bijkl,blk->bij", sigma_integrals, P_lk)
        G = scatter.add_tiles_2d(
            matrix=G,
            tiles=J_ij,
            row_starts=starts_i,
            col_starts=starts_j,
            mask=jnp.ones_like(batch_data.mask),
        )

        # Exchange update: G_il -= 0.5 * (ij|kl) * P_jk
        P_jk = gather.extract_tiles_2d(
            matrix=params.P,
            row_starts=starts_j,
            col_starts=starts_k,
            n_rows=n_j,
            n_cols=n_k,
        )
        K_il = jnp.einsum("bijkl,bjk->bil", sigma_integrals, P_jk)
        G = scatter.add_tiles_2d(
            matrix=G,
            tiles=-0.5 * K_il,
            row_starts=starts_i,
            col_starts=starts_l,
            mask=jnp.ones_like(batch_data.mask),
        )

    return G, None


def _process_batched_tuples(
    G: jax.Array,
    P: jax.Array,
    batched_tuples: batching.BatchedTreeTuples,
    global_block_starts: jax.Array,
    batch_operator: _BlockOperator,
) -> jax.Array:
    """Compute contributions to the two-electron matrix from batched tuples."""
    stack_starts = tuple(
        global_block_starts[idx] for idx in batched_tuples.global_tree_indices
    )
    params = _GroupParams(
        P, batched_tuples.stacks, stack_starts, batch_operator
    )
    batch_data = _BatchData(
        batched_tuples.tuple_indices, batched_tuples.padding_mask
    )
    scan_fn = functools.partial(_batch_step, params)
    new_G, _ = jax.lax.scan(scan_fn, G, batch_data)

    return new_G


def two_electron_matrix_jax(
    basis: bmb.BatchedMolecularBasis,
    P: jax.Array,
) -> jax.Array:
    """Compute the two-electron contribution to the Fock matrix.

    Args:
        basis: The batched molecular basis.
        P: The closed shell density matrix. shape: (n_basis, n_basis)
    Returns:
        The two-electron Fock matrix. Shape: (n_basis, n_basis).
    """
    n_basis = basis.basis.n_basis
    G = jnp.zeros((n_basis, n_basis), dtype=jnp.float64)
    batch_operator = jax.vmap(
        functools.partial(
            operators.two_electron_matrix, operator=coulomb.two_electron
        )
    )

    for batched_tuples in basis.batches_2e:
        G = _process_batched_tuples(
            G,
            P,
            batched_tuples,
            jnp.asarray(basis.block_starts),
            batch_operator,
        )

    return G


def two_electron_matrix(
    basis: bmb.BatchedMolecularBasis,
    P: np.ndarray,
) -> np.ndarray:
    return np.array(jit(two_electron_matrix_jax)(basis, jnp.array(P)))


def electronic_energy_jax(
    H_core: jax.Array, F: jax.Array, P: jax.Array
) -> jax.Array:
    """Compute the electronic expectation energy from the Fock and density
    matrices.

    Formula:
        E = 0.5 * sum_{ij}P_ij(H_core_ji + F_ji)

    Args:
        H_core: The core Hamiltonian matrix. shape: (n_basis, n_basis)
        F: The core Fock matrix. shape: (n_basis, n_basis)
        P: The closed shell density matrix. shape: (n_basis, n_basis)
    Returns:
        The electronic energy. Shape: ().
    """
    # Note that P is symmetric so we can use P_ij = P_ji
    return 0.5 * jnp.sum(P * (H_core + F))


def electronic_energy(
    H_core: np.ndarray, F: np.ndarray, P: np.ndarray
) -> float:
    return np.array(
        electronic_energy_jax(jnp.array(H_core), jnp.array(F), jnp.array(P))
    ).item()
