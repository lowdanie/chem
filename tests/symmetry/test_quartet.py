import pytest

from slaterform.hartree_fock import quartet as quartet_lib


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
    "quartet,expected_permutations",
    [
        (
            (1, 1, 1, 1),
            [
                (0, 1, 2, 3),
            ],
        ),
        (
            (1, 1, 2, 2),
            [
                (0, 1, 2, 3),
                (2, 3, 0, 1),
            ],
        ),
        (
            (1, 2, 2, 2),
            [
                (0, 1, 2, 3),
                (1, 0, 2, 3),
                (2, 3, 0, 1),
                (2, 3, 1, 0),
            ],
        ),
        (
            (1, 2, 3, 4),
            [
                (0, 1, 2, 3),
                (1, 0, 2, 3),
                (0, 1, 3, 2),
                (1, 0, 3, 2),
                (2, 3, 0, 1),
                (3, 2, 0, 1),
                (2, 3, 1, 0),
                (3, 2, 1, 0),
            ],
        ),
    ],
)
def test_generate_permutations(quartet, expected_permutations):
    actual_permutations = quartet_lib.generate_permutations(quartet)
    assert set(actual_permutations) == set(expected_permutations)


@pytest.mark.parametrize(
    "n",
    [1, 2, 3, 4, 5],
)
def test_generates_all_quartets(n):
    all_quartets = []
    for quartet in quartet_lib.iter_canonical_quartets(n):
        for sigma in quartet_lib.generate_permutations(quartet):
            all_quartets.append(quartet_lib.apply_permutation(sigma, quartet))

    expected_n_quartets = n**4
    assert len(all_quartets) == expected_n_quartets
    assert len(set(all_quartets)) == expected_n_quartets
    assert all(
        all(0 <= index < n for index in quartet) for quartet in all_quartets
    )
