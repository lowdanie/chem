from typing import Iterable, Literal
import itertools
import functools

import jax
from jax import numpy as jnp

# A quartet is a tuple of four indices which we will typically
# denote as (i, j, k, l).
Quartet = tuple[int, int, int, int]

# A batch of quartets. All of the arrays have the same shape.
BatchedQuartet = tuple[jax.Array, jax.Array, jax.Array, jax.Array]

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

# Symmetries of the two-electron integrals.
_SYMMETRIES: tuple[Permutation, ...] = (
    (0, 1, 2, 3),  # Identity
    (1, 0, 2, 3),  # Swap i-j
    (0, 1, 3, 2),  # Swap k-l
    (1, 0, 3, 2),  # Swap i-j, k-l
    (2, 3, 0, 1),  # Swap pairs (ij)-(kl)
    (3, 2, 0, 1),  # Swap pairs + Swap i-j (on new pos)
    (2, 3, 1, 0),  # Swap pairs + Swap k-l (on new pos)
    (3, 2, 1, 0),  # All swaps
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


def get_symmetries():
    return _SYMMETRIES


def apply_permutation(
    sigma: Permutation, quartet: BatchedQuartet
) -> BatchedQuartet:
    return (
        quartet[sigma[0]],
        quartet[sigma[1]],
        quartet[sigma[2]],
        quartet[sigma[3]],
    )


def compute_stabilizer_norm(quartet: BatchedQuartet) -> jax.Array:
    """Counts the number of symmetries that map the quartet onto itself.

    Args:
        quartet: A batched quartet of shape (batch_size, )

    Returns:
        A jax array of shape (batch_size, ) containing the count of
        symmetries mapping each quartet onto itself.
    """
    count = jax.numpy.zeros_like(quartet[0], dtype=jax.numpy.int32)

    for sigma in get_symmetries():
        permuted = apply_permutation(sigma, quartet)
        matches = tuple(permuted[i] == quartet[i] for i in range(4))
        is_stabilizer = functools.reduce(jnp.logical_and, matches)

        count += is_stabilizer

    return count
