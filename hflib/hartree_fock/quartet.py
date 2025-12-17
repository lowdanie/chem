from typing import Iterable, Literal
import itertools

# A quartet is a tuple of four indices which we will typically
# denote as (i, j, k, l).
Quartet = tuple[int, int, int, int]

# A permutation sigma is a tuple of unique indices in {0, 1, 2, 3}.
#
# The permutation sigma acts on a quartet q by "pulling":
# sigma(q) = (q[sigma[0]], q[sigma[1]], q[sigma[2]], q[sigma[3]])
#
# By definition, the composition of two permutations sigma1 and sigma2 is given
# by first applying sigma2, then sigma1:
# (sigma1 o sigma2)(q) = sigma1(sigma2(q))
#
# By this pulling definition, the composition of the permutations s1 and s2
# can be computed as:
# (s1 o s2) = (s2[s1[0]], s2[s1[1]], s2[s1[2]], s2[s1[3]])
#
# To prove this, note that:
# s1(s2(q)) = s1(q[s2[0]], q[s2[1]], q[s2[2]], q[s2[3]])
#           = (q[s2[s1[0]]], q[s2[s1[1]]], q[s2[s1[2]]], q[s2[s1[3]]
_Index = Literal[0, 1, 2, 3]
Permutation = tuple[_Index, _Index, _Index, _Index]

# (i,j,k,l) -> (j,i,k,l)
_SWAP_IJ: Permutation = (1, 0, 2, 3)

# (i,j,k,l) -> (i,j,l,k)
_SWAP_KL: Permutation = (0, 1, 3, 2)

# (i,j,k,l) -> (k,l,i,j)
_SWAP_IJ_KL: Permutation = (2, 3, 0, 1)


def _compose_permutations(
    sigma1: Permutation, sigma2: Permutation
) -> Permutation:
    """Returns the composition sigma1 o sigma2."""
    return (
        sigma2[sigma1[0]],
        sigma2[sigma1[1]],
        sigma2[sigma1[2]],
        sigma2[sigma1[3]],
    )


def _generate_permutations(
    eq_ij: bool, eq_kl: bool, eq_ij_kl: bool
) -> list[Permutation]:
    ops_ij = [False, True] if not eq_ij else [False]
    ops_kl = [False, True] if not eq_kl else [False]
    ops_ij_kl = [False, True] if not eq_ij_kl else [False]

    permutations = []
    for op_ij, op_kl, op_ij_kl in itertools.product(ops_ij, ops_kl, ops_ij_kl):
        sigma = (0, 1, 2, 3)
        if op_ij:
            sigma = _compose_permutations(_SWAP_IJ, sigma)
        if op_kl:
            sigma = _compose_permutations(_SWAP_KL, sigma)
        if op_ij_kl:
            sigma = _compose_permutations(_SWAP_IJ_KL, sigma)
        permutations.append(sigma)

    return permutations


def _generate_permutation_map() -> (
    dict[tuple[bool, bool, bool], list[Permutation]]
):
    permutation_map = {}
    for eq_ij, eq_kl, eq_ij_kl in itertools.product([False, True], repeat=3):
        key = (eq_ij, eq_kl, eq_ij_kl)
        permutation_map[key] = _generate_permutations(eq_ij, eq_kl, eq_ij_kl)

    return permutation_map


_PERMUTATION_MAP: dict[tuple[bool, bool, bool], list[Permutation]] = (
    _generate_permutation_map()
)


def iter_canonical_quartets(n: int) -> Iterable[Quartet]:
    """
    Yields unique tuples (i, j, k, l) satisfying:
    1. i >= j
    2. k >= l
    3. (i, j) >= (k, l)
    """
    # Create the list of all pairs (i, j) with i >= j
    pairs = [(i, j) for i in range(n) for j in range(i + 1)]

    # Iterate over pairs-of-pairs where (k, l) <= (i, j)
    for (k, l), (i, j) in itertools.combinations_with_replacement(pairs, 2):
        yield (i, j, k, l)


def apply_permutation(sigma: Permutation, quartet: Quartet) -> Quartet:
    return (
        quartet[sigma[0]],
        quartet[sigma[1]],
        quartet[sigma[2]],
        quartet[sigma[3]],
    )


def generate_permutations(quartet: Quartet) -> list[Permutation]:
    i, j, k, l = quartet
    eq_ij = i == j
    eq_kl = k == l
    eq_ij_kl = (i, j) == (k, l)

    return _PERMUTATION_MAP[(eq_ij, eq_kl, eq_ij_kl)]
