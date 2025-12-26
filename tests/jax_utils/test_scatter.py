import dataclasses
import pytest

import jax
from jax import numpy as jnp
import numpy as np

from slaterform.jax_utils import scatter


@dataclasses.dataclass
class _AddTiles2DTestCase:
    matrix: jax.Array
    tiles: jax.Array
    row_starts: jax.Array
    col_starts: jax.Array
    mask: jax.Array
    expected: jax.Array


@pytest.mark.parametrize(
    "case",
    [
        _AddTiles2DTestCase(
            matrix=jnp.ones((2, 2)),
            tiles=jnp.empty((0, 1, 1)),
            row_starts=jnp.empty((0,), dtype=jnp.int32),
            col_starts=jnp.empty((0,), dtype=jnp.int32),
            mask=jnp.empty((0,)),
            expected=jnp.ones((2, 2)),
        ),
        _AddTiles2DTestCase(
            matrix=jnp.ones((4, 4)),
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
            row_starts=jnp.array([0, 2, 0]),
            col_starts=jnp.array([0, 2, 2]),
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
        _AddTiles2DTestCase(
            matrix=jnp.zeros((4, 4)),
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
            row_starts=jnp.array([0, 0]),
            col_starts=jnp.array([0, 0]),
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
    ],
)
def test_add_tiles_2d(case: _AddTiles2DTestCase):
    actual = scatter.add_tiles_2d(
        matrix=case.matrix,
        tiles=case.tiles,
        row_starts=case.row_starts,
        col_starts=case.col_starts,
        mask=case.mask,
    )
    np.testing.assert_allclose(actual, case.expected)
