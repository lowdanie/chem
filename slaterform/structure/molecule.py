from collections.abc import Sequence
import dataclasses

from jax.tree_util import register_pytree_node_class

from slaterform.structure import atom as atom_lib
from slaterform.adapters import bse


@register_pytree_node_class
@dataclasses.dataclass
class Molecule:
    atoms: Sequence[atom_lib.Atom]

    def tree_flatten(self):
        children = (self.atoms,)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(
        cls, aux_data: None, children: tuple[Sequence[atom_lib.Atom],]
    ) -> "Molecule":
        return cls(atoms=children[0])

    @classmethod
    def from_geometry(
        cls, atoms: Sequence[atom_lib.Atom], basis_name: str
    ) -> "Molecule":
        """Builds a Molecule object from a sequence of atomic positions and a basis set name."""
        atoms = [
            atom_lib.Atom(
                symbol=atom.symbol,
                number=atom.number,
                position=atom.position,
                shells=bse.load(basis_name=basis_name, element=atom.number),
            )
            for atom in atoms
        ]

        return cls(atoms=atoms)
