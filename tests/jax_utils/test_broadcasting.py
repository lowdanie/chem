import pytest
import dataclasses
import itertools

import numpy as np

from slaterform.jax_utils import broadcasting


@dataclasses.dataclass
class _BrodcastIndicesTestCase:
    index_blocks: list[np.ndarray]
    expected: tuple[np.ndarray, ...]


@dataclasses.dataclass
class _FlatProductTestCase:
    arrays: list[np.ndarray]


@pytest.mark.parametrize(
    "case",
    [
        _BrodcastIndicesTestCase(
            index_blocks=[
                np.array(
                    [
                        [0, 0, 0],
                    ]
                )
            ],
            expected=(np.array([0]), np.array([0]), np.array([0])),
        ),
        _BrodcastIndicesTestCase(
            index_blocks=[
                np.array(
                    [
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                    ]
                )
            ],
            expected=(
                np.array([0, 1, 0, 0]),
                np.array([0, 0, 1, 0]),
                np.array([0, 0, 0, 1]),
            ),
        ),
        _BrodcastIndicesTestCase(
            index_blocks=[
                np.array(
                    [
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                    ]
                ),
                np.array(
                    [
                        [0, 0, 0],
                        [0, 1, 0],
                    ]
                ),
            ],
            expected=(
                np.array([[0], [1], [0], [0]]),
                np.array([[0], [0], [1], [0]]),
                np.array([[0], [0], [0], [1]]),
                np.array([[0, 0]]),
                np.array([[0, 1]]),
                np.array([[0, 0]]),
            ),
        ),
        _BrodcastIndicesTestCase(
            index_blocks=[
                np.array(
                    [
                        [0, 0, 0],
                        [1, 0, 0],
                    ]
                ),
                np.array(
                    [
                        [1, 0, 0],
                        [0, 1, 0],
                    ]
                ),
                np.array(
                    [
                        [0, 1, 0],
                        [0, 0, 1],
                    ]
                ),
                np.array(
                    [
                        [0, 0, 1],
                        [0, 0, 0],
                    ]
                ),
            ],
            expected=(
                np.array([[[[0]]], [[[1]]]]),
                np.array([[[[0]]], [[[0]]]]),
                np.array([[[[0]]], [[[0]]]]),
                np.array([[[[1]], [[0]]]]),
                np.array([[[[0]], [[1]]]]),
                np.array([[[[0]], [[0]]]]),
                np.array([[[[0], [0]]]]),
                np.array([[[[1], [0]]]]),
                np.array([[[[0], [1]]]]),
                np.array([[[[0, 0]]]]),
                np.array([[[[0, 0]]]]),
                np.array([[[[1, 0]]]]),
            ),
        ),
    ],
)
def test_broadcast_indices(case):
    result = broadcasting.broadcast_indices(*case.index_blocks)
    for r, e in zip(result, case.expected):
        np.testing.assert_array_equal(r, e)


def test_broadcast_to_2d():
    block1 = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ]
    )
    block2 = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
        ]
    )

    T = np.zeros((2, 2, 2, 2, 2, 2))
    T[0, 0, 0, 1, 0, 0] = 1.0
    T[0, 0, 0, 0, 1, 0] = 2.0
    T[1, 0, 0, 1, 0, 0] = 3.0
    T[1, 0, 0, 0, 1, 0] = 4.0
    T[0, 1, 0, 1, 0, 0] = 5.0
    T[0, 1, 0, 0, 1, 0] = 6.0

    indices = broadcasting.broadcast_indices(block1, block2)
    M = T[indices]

    expected = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )

    np.testing.assert_array_equal(M, expected)


def test_broadcast_to_4d():
    b1 = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
        ]
    )
    b2 = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
        ]
    )
    b3 = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    b4 = np.array(
        [
            [0, 0, 1],
            [0, 0, 0],
        ]
    )

    T = np.zeros((2,) * 12)

    #    b1  |    b2   |   b3   |   b4
    T[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1] = 1.0
    T[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0] = 2.0
    T[0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1] = 3.0
    T[0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0] = 4.0

    T[0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1] = 5.0
    T[0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0] = 6.0
    T[0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1] = 7.0
    T[0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0] = 8.0

    T[1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1] = 9.0
    T[1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0] = 10.0
    T[1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1] = 11.0
    T[1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0] = 12.0

    T[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1] = 13.0
    T[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0] = 14.0
    T[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1] = 15.0
    T[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0] = 16.0

    indices = broadcasting.broadcast_indices(b1, b2, b3, b4)

    expected = np.zeros((2, 2, 2, 2))
    expected[0, 0, 0, 0] = 1.0
    expected[0, 0, 0, 1] = 2.0
    expected[0, 0, 1, 0] = 3.0
    expected[0, 0, 1, 1] = 4.0
    expected[0, 1, 0, 0] = 5.0
    expected[0, 1, 0, 1] = 6.0
    expected[0, 1, 1, 0] = 7.0
    expected[0, 1, 1, 1] = 8.0
    expected[1, 0, 0, 0] = 9.0
    expected[1, 0, 0, 1] = 10.0
    expected[1, 0, 1, 0] = 11.0
    expected[1, 0, 1, 1] = 12.0
    expected[1, 1, 0, 0] = 13.0
    expected[1, 1, 0, 1] = 14.0
    expected[1, 1, 1, 0] = 15.0
    expected[1, 1, 1, 1] = 16.0

    M = T[indices]
    np.testing.assert_array_equal(M, expected)


@pytest.mark.parametrize(
    "case",
    [
        _FlatProductTestCase(
            arrays=[
                np.array([1, 2]),
                np.array([3, 4]),
            ],
        ),
        _FlatProductTestCase(
            arrays=[
                np.array([1, 2]),
                np.array([3, 4]),
                np.array([5, 6]),
                np.array([7, 8]),
            ],
        ),
    ],
)
def test_flat_product(case):
    result = broadcasting.flat_product(*case.arrays)
    result = [r.tolist() for r in result]
    pairs = list(zip(*result))

    expected_pairs = set(itertools.product(*case.arrays))

    assert len(pairs) == len(expected_pairs)
    assert set(pairs) == expected_pairs
