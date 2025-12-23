from collections.abc import Sequence
import dataclasses

from jax.tree_util import register_pytree_node_class

from slaterform.structure import atom


@register_pytree_node_class
@dataclasses.dataclass
class Molecule:
    atoms: Sequence[atom.Atom]

    def tree_flatten(self):
        children = (self.atoms,)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(
        cls, aux_data: None, children: tuple[Sequence[atom.Atom]]
    ) -> "Molecule":
        return cls(atoms=children[0])
