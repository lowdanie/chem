import dataclasses
import itertools
from collections.abc import Sequence

import jax
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class
import numpy as np

from slaterform.basis import basis_block
from slaterform.jax_utils import batching
from slaterform.structure import molecular_basis
from slaterform.symmetry import quartet
from slaterform import types


@register_pytree_node_class
@dataclasses.dataclass
class BatchedMolecularBasis:
    """A MolecularBasis with pre-computed batch structures for integration.

    Holds both 1-electron (pair) and 2-electron batches.
    """

    basis: molecular_basis.MolecularBasis

    # The starting indices of each basis block in the full basis set.
    # Shape: (n_blocks,)
    block_starts: jax.Array

    # Batches for 1-electron integrals (H_core, S). Tuple length = 2.
    batches_1e: Sequence[batching.BatchedTreeTuples]

    # Batches for 2-electron integrals. Tuple length = 4.
    batches_2e: Sequence[batching.BatchedTreeTuples]

    def tree_flatten(self):
        children = (
            self.basis,
            self.block_starts,
            self.batches_1e,
            self.batches_2e,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            basis=children[0],
            block_starts=children[1],
            batches_1e=children[2],
            batches_2e=children[3],
        )


def batch_basis(
    basis: molecular_basis.MolecularBasis,
    batch_size_1e: int = 4096,
    batch_size_2e: int = 2048,
) -> BatchedMolecularBasis:
    n_blocks = len(basis.basis_blocks)
    block_sizes = np.array([block.n_basis for block in basis.basis_blocks])
    block_starts = block_starts = jnp.array(
        np.concatenate(([0], np.cumsum(block_sizes)[:-1])), dtype=jnp.int32
    )

    # (i, j) where 0 <= i <= j < n_blocks
    pairs = list(itertools.combinations_with_replacement(range(n_blocks), 2))
    batches_1e = batching.batch_tree_tuples(
        trees=basis.basis_blocks,
        tuple_length=2,
        tuple_indices=pairs,
        max_batch_size=batch_size_1e,
    )

    batches_2e = batching.batch_tree_tuples(
        trees=basis.basis_blocks,
        tuple_length=4,
        tuple_indices=list(quartet.iter_canonical_quartets(n_blocks)),
        max_batch_size=batch_size_2e,
    )

    return BatchedMolecularBasis(basis, block_starts, batches_1e, batches_2e)
