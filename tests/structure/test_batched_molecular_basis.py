from collections import Counter
import dataclasses
import pytest

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class
from unittest import mock

import slaterform as sf
from tests.jax_utils import block_utils


@register_pytree_node_class
@dataclasses.dataclass
class MockBlock:
    n_basis: int
    data: jax.Array

    def tree_flatten(self):
        return (self.data,), self.n_basis

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(n_basis=aux, data=children[0])


def build_basis(block_sizes: list[int]) -> sf.MolecularBasis:
    blocks = [
        MockBlock(n_basis=block_size, data=jnp.ones((1,)))
        for block_size in block_sizes
    ]

    basis = mock.Mock(spec=sf.MolecularBasis)
    basis.basis_blocks = blocks

    return basis


@pytest.mark.parametrize(
    "block_sizes,expected",
    [
        (
            [10],
            [0],
        ),
        (
            [10, 5],
            [0, 10],
        ),
        (
            [3, 7, 2],
            [0, 3, 10],
        ),
    ],
)
def test_block_starts(block_sizes, expected):
    basis = build_basis(block_sizes)
    batched = sf.structure.batch_basis(basis)
    np.testing.assert_array_equal(batched.block_starts, expected)


@pytest.mark.parametrize(
    "block_sizes,expected",
    [
        (
            [2, 3],
            [
                (0, 0),
                (0, 1),
                (1, 1),
            ],
        ),
        (
            [2, 3, 4],
            [
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 1),
                (1, 2),
                (2, 2),
            ],
        ),
    ],
)
def test_1e_tuples(block_sizes, expected):
    """Verifies that the batches contain stacked JAX arrays."""
    basis = build_basis(block_sizes)
    batched = sf.structure.batch_basis(basis, batch_size_1e=2)

    actual = []
    for batched_tuples in batched.batches_1e:
        assert batched_tuples.tuple_indices.shape[0] <= 2  # batch size
        actual.extend(block_utils.get_global_tuple_indices(batched_tuples))

    assert Counter(actual) == Counter(expected)


@pytest.mark.parametrize(
    "block_sizes,expected",
    [
        (
            [2, 3],
            [
                (0, 0, 0, 0),
                (1, 0, 0, 0),
                (1, 0, 1, 0),
                (1, 1, 0, 0),
                (1, 1, 1, 0),
                (1, 1, 1, 1),
            ],
        ),
    ],
)
def test_2e_tuples(block_sizes, expected):
    basis = build_basis(block_sizes)
    batched = sf.structure.batch_basis(basis, batch_size_2e=2)

    actual = []
    for batched_tuples in batched.batches_2e:
        assert batched_tuples.tuple_indices.shape[0] <= 2  # batch size
        actual.extend(block_utils.get_global_tuple_indices(batched_tuples))

    print("actual")
    for t in actual:
        print(t)
    assert Counter(actual) == Counter(expected)
