import numpy as np

from integrals import overlap
from integrals import gaussian


def _generate_cartesian_powers(l: int) -> np.ndarray:
    """Generates the Cartesian powers for the given angular momentum.

    Returns:
        A numpy array of shape (M, 3) where M is the total number of
        triples (i, j, k) of non-negative integers satisfying:
        i + j + k = l.
    """
    powers = [
        (i, j, l - i - j)
        for i in range(l, -1, -1)
        for j in range(l - i, -1, -1)
    ]
    return np.array(powers, dtype=np.int32)


_MAX_PRECOMPUTE_L = 1
_CARTESIAN_POWERS_CACHE: tuple[np.ndarray, ...] = tuple(
    _generate_cartesian_powers(l) for l in range(_MAX_PRECOMPUTE_L + 1)
)


def generate_cartesian_powers(l: int) -> np.ndarray:
    """Generates the Cartesian powers for the given angular momentum.

    Returns:
        A numpy array of shape (M, 3) where M is the total number of
        triples (i, j, k) of non-negative integers satisfying:
        i + j + k = l.
    """
    if l > _MAX_PRECOMPUTE_L:
        raise ValueError(
            f"Angular momentum {l} not precomputed. "
            f"Max precomputed is {_MAX_PRECOMPUTE_L}."
        )

    return _CARTESIAN_POWERS_CACHE[l]


def compute_normalization_constants(
    a: float, max_degree: int, powers: np.ndarray
) -> np.ndarray:
    """Computes the L^2 norms of the primitive Cartesian Gaussians

    Args:
        a: The exponent of the Gaussian primitives.
        max_degree: The maximum angular momentum degree.
        powers: A numpy array of shape (N, 3) containing the Cartesian
                powers (i, j, k) for each basis function.

    Returns:
        A numpy array of shape (N,) containing the L^2 norms of each
        primitive Gaussian basis function.
    """
    g = gaussian.GaussianBasis3d(
        max_degree=max_degree,
        exponent=a,
        center=np.zeros(3),
    )

    S = overlap.overlap_3d(g, g)  # shape (max_degree+1,) * 6
    ix, iy, iz = powers.T
    return 1.0 / np.sqrt(S[ix, iy, iz, ix, iy, iz])
