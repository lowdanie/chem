import dataclasses
from typing import Callable
from collections.abc import Sequence

import numpy as np

from slaterform.structure import atom
from slaterform.structure import molecule
from slaterform.basis import basis_block
from slaterform.basis import contracted_gto

# Fetches a list of contracted GTOs for a given atomic number.
BasisFetcher = Callable[[int], Sequence[contracted_gto.ContractedGTO]]


@dataclasses.dataclass
class MolecularBasis:
    atoms: list[atom.Atom]
    basis_blocks: list[basis_block.BasisBlock]

    # The start and end indices of each basis block in the full basis set.
    # length: len(basis_blocks)
    block_slices: list[slice]

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

    block_sizes = np.array(
        [block.n_basis for block in basis_blocks], dtype=np.int32
    )
    block_starts = np.concatenate(([0], np.cumsum(block_sizes)))
    block_slices = [
        slice(start, start + size)
        for start, size in zip(block_starts, block_sizes)
    ]

    return MolecularBasis(
        atoms=molecule.atoms,
        basis_blocks=basis_blocks,
        block_slices=block_slices,
    )
