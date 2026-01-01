import jax
import jax.numpy as jnp
import numpy as np

import slaterform.types as types
from slaterform.integrals.overlap import overlap_3d
from slaterform.integrals.gaussian import GaussianBasis3d


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
    max_degree: int, powers: types.StaticArray, a: types.Array
) -> jax.Array:
    """Computes the L^2 norms of the primitive Cartesian Gaussians

    Args:
        max_degree: The maximum angular momentum degree.
        powers: A numpy array of shape (N, 3) containing the Cartesian
                powers (i, j, k) for each basis function.
        a: The exponent of the Gaussian primitives. Shape ()


    Returns:
        A numpy array of shape (N,) containing the L^2 norms of each
        primitive Gaussian basis function.
    """
    g = GaussianBasis3d(
        max_degree=max_degree,
        exponent=a,
        center=jnp.zeros(3, dtype=a.dtype),
    )

    S = overlap_3d(g, g)  # shape (max_degree+1,) * 6
    ix, iy, iz = powers.T
    return 1.0 / jnp.sqrt(S[ix, iy, iz, ix, iy, iz])
