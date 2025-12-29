import collections
import pytest

import jax
from jax import numpy as jnp
import numpy as np

from slaterform.symmetry import quartet as quartet_lib


@pytest.mark.parametrize(
    "n,expected",
    [
        (0, set()),
        (
            1,
            {
                (0, 0, 0, 0),
            },
        ),
        (
            2,
            {
                (0, 0, 0, 0),
                (1, 0, 0, 0),
                (1, 0, 1, 0),
                (1, 1, 0, 0),
                (1, 1, 1, 0),
                (1, 1, 1, 1),
            },
        ),
    ],
)
def test_iter_canonical_quartets(n, expected):
    actual = set(quartet_lib.iter_canonical_quartets(n))
    assert actual == expected


def test_get_symmetries():
    assert quartet_lib.get_symmetries() == quartet_lib._SYMMETRIES


@pytest.mark.parametrize(
    "quartet,permutation,expected",
    [
        ((4, 5, 6, 7), (0, 1, 2, 3), (4, 5, 6, 7)),
        ((4, 5, 6, 7), (1, 0, 2, 3), (5, 4, 6, 7)),
        ((4, 5, 6, 7), (0, 1, 3, 2), (4, 5, 7, 6)),
        ((4, 5, 6, 7), (3, 2, 0, 1), (7, 6, 4, 5)),
    ],
)
def test_apply_permutation(quartet, permutation, expected):
    actual = quartet_lib.apply_permutation(permutation, quartet)
    assert actual == expected


@pytest.mark.parametrize(
    "quartet,permutation,expected",
    [
        (
            (
                jnp.array([4, 1]),
                jnp.array([5, 2]),
                jnp.array([6, 3]),
                jnp.array([7, 4]),
            ),
            (0, 1, 2, 3),
            (
                jnp.array([4, 1]),
                jnp.array([5, 2]),
                jnp.array([6, 3]),
                jnp.array([7, 4]),
            ),
        ),
        (
            (
                jnp.array([4, 1]),
                jnp.array([5, 2]),
                jnp.array([6, 3]),
                jnp.array([7, 4]),
            ),
            (0, 1, 3, 2),
            (
                jnp.array([4, 1]),
                jnp.array([5, 2]),
                jnp.array([7, 4]),
                jnp.array([6, 3]),
            ),
        ),
        (
            (
                jnp.array([4, 1]),
                jnp.array([5, 2]),
                jnp.array([6, 3]),
                jnp.array([7, 4]),
            ),
            (1, 0, 2, 3),
            (
                jnp.array([5, 2]),
                jnp.array([4, 1]),
                jnp.array([6, 3]),
                jnp.array([7, 4]),
            ),
        ),
    ],
)
def test_apply_permutation_batch(quartet, permutation, expected):
    actual = quartet_lib.apply_permutation(permutation, quartet)
    for a, e in zip(actual, expected):
        np.testing.assert_array_equal(a, e)


@pytest.mark.parametrize(
    "n",
    [1, 2, 3, 4, 5],
)
def test_generates_all_quartets(n):
    quartet_counts = collections.defaultdict(float)

    for quartet in quartet_lib.iter_canonical_quartets(n):
        inv_stabilizer_norm = 1.0 / quartet_lib.compute_stabilizer_norm(quartet)
        for sigma in quartet_lib.get_symmetries():
            sigma_quartet = quartet_lib.apply_permutation(sigma, quartet)
            quartet_counts[sigma_quartet] += inv_stabilizer_norm

    for quartet, count in quartet_counts.items():
        print(f"{quartet}: {count}")

    expected_n_quartets = n**4
    assert len(quartet_counts) == expected_n_quartets
    assert len(set(quartet_counts)) == expected_n_quartets

    for quartet, count in quartet_counts.items():
        assert all(0 <= index < n for index in quartet)
        np.testing.assert_allclose(count, 1.0, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize(
    "case",
    [
        [
            ((0, 0, 0, 0), 8),
            ((0, 0, 0, 1), 2),
            ((0, 0, 1, 0), 2),
            ((0, 0, 1, 1), 4),
            ((0, 1, 0, 0), 2),
            ((0, 1, 0, 1), 2),
            ((0, 1, 1, 0), 2),
            ((0, 1, 1, 1), 2),
            ((1, 0, 0, 0), 2),
            ((1, 0, 0, 1), 2),
            ((1, 0, 1, 0), 2),
            ((1, 0, 1, 1), 2),
            ((1, 1, 0, 0), 4),
            ((1, 1, 0, 1), 2),
            ((1, 1, 1, 0), 2),
            ((1, 1, 1, 1), 8),
        ],
        [
            ((0, 1, 2, 3), 1),
            ((0, 1, 2, 1), 1),
            ((0, 1, 1, 2), 1),
            ((2, 2, 3, 3), 4),
            ((0, 1, 2, 0), 1),
            ((0, 1, 2, 2), 2),
        ],
    ],
)
def test_compute_stabilizer_norm(case):
    quartets = (
        jnp.array([q[0] for q, _ in case]),
        jnp.array([q[1] for q, _ in case]),
        jnp.array([q[2] for q, _ in case]),
        jnp.array([q[3] for q, _ in case]),
    )
    expected = jnp.array([count for _, count in case])

    actual = quartet_lib.compute_stabilizer_norm(quartets)
    np.testing.assert_array_equal(actual, expected)
