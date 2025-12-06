import functools

import numpy as np


@functools.lru_cache(maxsize=None)
def generate_grlex_triples(max_degree: int) -> np.ndarray:
    """
    Generates exponent triples (i, j, k) in graded lexicographic order.

    Args:
        max_degree: The maximum total degree D = i + j + k.
    Returns:
        A numpy array of shape (N, 3) where N is the number of triples with
        total degree up to max_degree.
    """
    return np.array(
        [
            (i, j, D - i - j)
            for D in range(max_degree + 1)
            for i in range(D, -1, -1)
            for j in range(D - i, -1, -1)
        ]
    )
