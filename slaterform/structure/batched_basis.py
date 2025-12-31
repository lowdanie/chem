import dataclasses
import itertools
from collections.abc import Sequence

import jax
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class
import numpy as np

from slaterform.basis import basis_block
from slaterform.jax_utils import batching
from slaterform.structure import atom as atom_lib
from slaterform.structure import molecule as mol_lib
from slaterform.symmetry import quartet


def _build_basis_blocks(
    atoms: Sequence[atom_lib.Atom],
) -> Sequence[basis_block.BasisBlock]:
    basis_blocks = []
    for atom in atoms:
        basis_blocks.extend(
            basis_block.build_basis_block(gto, atom.position)
            for gto in atom.shells
        )

    return basis_blocks


@register_pytree_node_class
@dataclasses.dataclass
class BatchedBasis:
    """A Molecule with pre-computed batched basis structures for integration.

    Holds both 1-electron (pair) and 2-electron batches.
    """

    atoms: Sequence[atom_lib.Atom]
    basis_blocks: Sequence[basis_block.BasisBlock]

    # The starting indices of each basis block in the full basis set.
    # Shape: (n_blocks,)
    block_starts: jax.Array

    # Batches for 1-electron integrals. Tuple length = 2.
    batches_1e: Sequence[batching.BatchedTreeTuples]

    # Batches for 2-electron integrals. Tuple length = 4.
    batches_2e: Sequence[batching.BatchedTreeTuples]

    @property
    def n_basis(self) -> int:
        """The total number of basis functions in this molecular basis."""
        return sum(block.n_basis for block in self.basis_blocks)

    @property
    def n_electrons(self) -> int:
        """The total number of electrons in the molecule."""
        return sum(atom.number for atom in self.atoms)

    def tree_flatten(self):
        children = (
            self.atoms,
            self.basis_blocks,
            self.block_starts,
            self.batches_1e,
            self.batches_2e,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            atoms=children[0],
            basis_blocks=children[1],
            block_starts=children[2],
            batches_1e=children[3],
            batches_2e=children[4],
        )

    @classmethod
    def from_molecule(
        cls,
        molecule: mol_lib.Molecule,
        batch_size_1e: int = 4096,
        batch_size_2e: int = 2048,
    ) -> "BatchedBasis":
        basis_blocks = _build_basis_blocks(molecule.atoms)

        n_blocks = len(basis_blocks)
        block_sizes = np.array([block.n_basis for block in basis_blocks])
        block_starts = jnp.array(
            np.concatenate(([0], np.cumsum(block_sizes)[:-1])), dtype=jnp.int32
        )

        # (i, j) where 0 <= i <= j < n_blocks
        pairs = list(
            itertools.combinations_with_replacement(range(n_blocks), 2)
        )
        batches_1e = batching.batch_tree_tuples(
            trees=basis_blocks,
            tuple_length=2,
            tuple_indices=pairs,
            max_batch_size=batch_size_1e,
        )

        batches_2e = batching.batch_tree_tuples(
            trees=basis_blocks,
            tuple_length=4,
            tuple_indices=list(quartet.iter_canonical_quartets(n_blocks)),
            max_batch_size=batch_size_2e,
        )

        return cls(
            molecule.atoms, basis_blocks, block_starts, batches_1e, batches_2e
        )
