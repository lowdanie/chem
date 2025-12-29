import dataclasses
import pytest

import jax
from jax import numpy as jnp
import numpy as np

from slaterform.jax_utils import gather


@dataclasses.dataclass
class _ExtractTilesTestCase:
    matrix: jax.Array
    row_starts: jax.Array
    col_starts: jax.Array
    n_rows: int
    n_cols: int
    expected_tiles: jax.Array


@pytest.mark.parametrize(
    "case",
    [
        _ExtractTilesTestCase(
            matrix=jnp.array(
                [
                    [11, 12, 13, 14],
                    [15, 16, 17, 18],
                    [19, 20, 21, 22],
                    [23, 24, 25, 26],
                ]
            ),
            row_starts=jnp.array([0, 0, 1, 2]),
            col_starts=jnp.array([0, 1, 1, 2]),
            n_rows=2,
            n_cols=2,
            expected_tiles=jnp.array(
                [
                    [[11, 12], [15, 16]],
                    [[12, 13], [16, 17]],
                    [[16, 17], [20, 21]],
                    [[21, 22], [25, 26]],
                ]
            ),
        ),
        _ExtractTilesTestCase(
            matrix=jnp.array(
                [
                    [11, 12, 13, 14],
                    [15, 16, 17, 18],
                    [19, 20, 21, 22],
                    [23, 24, 25, 26],
                ]
            ),
            row_starts=jnp.array([0, 1, 2, 3]),
            col_starts=jnp.array([0, 1, 1, 2]),
            n_rows=1,
            n_cols=2,
            expected_tiles=jnp.array(
                [
                    [[11, 12]],
                    [[16, 17]],
                    [[20, 21]],
                    [[25, 26]],
                ]
            ),
        ),
    ],
)
def test_extract_tiles_2d(case: _ExtractTilesTestCase):
    actual = gather.extract_tiles_2d(
        matrix=case.matrix,
        row_starts=case.row_starts,
        col_starts=case.col_starts,
        n_rows=case.n_rows,
        n_cols=case.n_cols,
    )
    np.testing.assert_array_equal(actual, case.expected_tiles)
