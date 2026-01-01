import dataclasses
import pytest
from collections.abc import Sequence

import jax
from jax import numpy as jnp
import numpy as np

from slaterform.jax_utils import scatter


@dataclasses.dataclass
class _AddTilesTestCase:
    target: jax.Array
    tiles: jax.Array
    starts: Sequence[jax.Array]
    mask: jax.Array
    expected: jax.Array


@pytest.mark.parametrize(
    "case",
    [
        _AddTilesTestCase(
            target=jnp.ones((2, 2)),
            tiles=jnp.empty((0, 1, 1)),
            starts=[
                jnp.empty((0,), dtype=jnp.int32),
                jnp.empty((0,), dtype=jnp.int32),
            ],
            mask=jnp.empty((0,)),
            expected=jnp.ones((2, 2)),
        ),
        _AddTilesTestCase(
            target=jnp.array([1, 2, 3, 4]),
            tiles=jnp.array([[10], [20], [30]]),
            starts=[jnp.array([0, 2, 3])],
            mask=jnp.array([1, 0, 1]),
            expected=jnp.array([11, 2, 3, 34]),
        ),
        _AddTilesTestCase(
            target=jnp.ones((4, 4)),
            tiles=jnp.array(
                [
                    [
                        [1, 1],
                        [1, 1],
                    ],
                    [
                        [2, 2],
                        [2, 2],
                    ],
                    [
                        [3, 3],
                        [3, 3],
                    ],
                ]
            ),
            starts=[
                jnp.array([0, 2, 0]),
                jnp.array([0, 2, 2]),
            ],
            mask=jnp.array([1, 1, 0]),
            expected=jnp.array(
                [
                    [2, 2, 1, 1],
                    [2, 2, 1, 1],
                    [1, 1, 3, 3],
                    [1, 1, 3, 3],
                ]
            ),
        ),
        _AddTilesTestCase(
            target=jnp.zeros((4, 4)),
            tiles=jnp.array(
                [
                    [
                        [1, 1],
                        [1, 1],
                    ],
                    [
                        [2, 2],
                        [2, 2],
                    ],
                ]
            ),
            starts=[
                jnp.array([0, 0]),
                jnp.array([0, 0]),
            ],
            mask=jnp.array([1, 1]),
            expected=jnp.array(
                [
                    [3, 3, 0, 0],
                    [3, 3, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ]
            ),
        ),
        _AddTilesTestCase(
            target=jnp.ones((4, 4, 4)),
            tiles=jnp.array([2 * np.ones((2, 2, 2)), 3 * np.ones((2, 2, 2))]),
            starts=[
                jnp.array([0, 1]),
                jnp.array([0, 0]),
                jnp.array([0, 2]),
            ],
            mask=jnp.array([1, 1]),
            expected=jnp.array(
                [
                    [
                        [3, 3, 1, 1],
                        [3, 3, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                    ],
                    [
                        [3, 3, 4, 4],
                        [3, 3, 4, 4],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                    ],
                    [
                        [1, 1, 4, 4],
                        [1, 1, 4, 4],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                    ],
                    [
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                    ],
                ]
            ),
        ),
    ],
)
def test_add_tiles(case: _AddTilesTestCase):
    actual = scatter.add_tiles(
        target=case.target,
        tiles=case.tiles,
        starts=case.starts,
        mask=case.mask,
    )
    print(actual)
    np.testing.assert_allclose(actual, case.expected)
