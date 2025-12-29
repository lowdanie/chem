import dataclasses

import jax
from jax.tree_util import register_pytree_node_class

from slaterform import types


@register_pytree_node_class
@dataclasses.dataclass
class Atom:
    symbol: str
    number: int  # Atomic number

    # Position in Bohr units
    position: types.Array

    def __post_init__(self):
        types.promote_dataclass_fields(self)

    def tree_flatten(self):
        children = (self.position,)
        aux_data = (self.symbol, self.number)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(
        cls, aux_data: tuple[str, int], children: tuple[jax.Array, ...]
    ) -> "Atom":
        return cls(
            symbol=aux_data[0],
            number=aux_data[1],
            position=children[0],
        )
