import dataclasses

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
import numpy as np

import slaterform.types as types


@register_pytree_node_class
@dataclasses.dataclass
class GaussianBasis1d:
    max_degree: int
    exponent: types.Array  # shape ()
    center: types.Array  # shape ()

    def __post_init__(self):
        types.promote_dataclass_fields(self)

    def tree_flatten(self):
        children = (self.exponent, self.center)
        aux_data = self.max_degree
        return (children, aux_data)

    @classmethod
    def tree_unflatten(
        cls, aux_data: int, children: tuple[jax.Array, jax.Array]
    ) -> "GaussianBasis1d":
        return cls(
            max_degree=aux_data, exponent=children[0], center=children[1]
        )


@register_pytree_node_class
@dataclasses.dataclass
class GaussianBasis3d:
    max_degree: int
    exponent: types.Array  # shape ()
    center: types.Array  # shape (3,)

    def __post_init__(self):
        types.promote_dataclass_fields(self)

    def tree_flatten(self):
        children = (self.exponent, self.center)
        aux_data = self.max_degree
        return (children, aux_data)

    @classmethod
    def tree_unflatten(
        cls, aux_data: int, children: tuple[jax.Array, jax.Array]
    ) -> "GaussianBasis3d":
        return cls(
            max_degree=aux_data, exponent=children[0], center=children[1]
        )


def gaussian_3d_to_1d(g: GaussianBasis3d, dim: int) -> GaussianBasis1d:
    center = jnp.asarray(g.center)
    return GaussianBasis1d(
        max_degree=g.max_degree, exponent=g.exponent, center=center[dim]
    )


def overlap_prefactor_1d(g1: GaussianBasis1d, g2: GaussianBasis1d) -> jax.Array:
    mu = (g1.exponent * g2.exponent) / (g1.exponent + g2.exponent)
    diff = g1.center - g2.center
    return jnp.exp(-mu * jnp.square(diff))


def overlap_prefactor_3d(g1: GaussianBasis3d, g2: GaussianBasis3d) -> jax.Array:
    mu = (g1.exponent * g2.exponent) / (g1.exponent + g2.exponent)
    diff = g1.center - g2.center
    return jnp.exp(-mu * jnp.dot(diff, diff))
