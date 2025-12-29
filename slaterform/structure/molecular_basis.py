import dataclasses
from typing import Callable
from collections.abc import Sequence

import jax
from jax import jit
from jax.tree_util import register_pytree_node_class
import numpy as np

from slaterform.structure import atom
from slaterform.structure import molecule
from slaterform.basis import basis_block
from slaterform.basis import contracted_gto

# Fetches a list of contracted GTOs for a given atomic number.
BasisFetcher = Callable[[int], Sequence[contracted_gto.ContractedGTO]]


@register_pytree_node_class
@dataclasses.dataclass
class MolecularBasis:
    atoms: Sequence[atom.Atom]
    basis_blocks: Sequence[basis_block.BasisBlock]

    @property
    def molecule(self) -> molecule.Molecule:
        """The molecule associated with this molecular basis."""
        return molecule.Molecule(atoms=self.atoms)

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
        )
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(
        cls,
        aux_data,
        children: tuple[
            Sequence[atom.Atom],
            Sequence[basis_block.BasisBlock],
        ],
    ) -> "MolecularBasis":
        return cls(
            atoms=children[0],
            basis_blocks=children[1],
        )


def build(
    molecule: molecule.Molecule, basis_fetcher: BasisFetcher
) -> MolecularBasis:
    basis_blocks = []
    for atom in molecule.atoms:
        contracted_gtos = basis_fetcher(atom.number)
        basis_blocks.extend(
            basis_block.build_basis_block(gto, atom.position)
            for gto in contracted_gtos
        )

    return MolecularBasis(
        atoms=molecule.atoms,
        basis_blocks=basis_blocks,
    )
