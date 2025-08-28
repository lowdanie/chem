import numpy as np

from integrals import gaussian


def kinetic_1d_from_overlap_1d(
    S: np.ndarray,
    g1: gaussian.GaussianBasis1d,
    g2: gaussian.GaussianBasis1d,
) -> np.ndarray:
    """
    We assume that S = overlap_1d(g1, g2)
    The output T has shape (S.shape[0], S.shape[1] - 2).

    Set:
    d1 = g1.max_degree
    a = g1.exponent
    A = g1.center
    d2 = g2.max_degree
    b = g2.exponent
    B = g2.center

    For each 0 <= i < d1 and 0 <= j <d2 - 2:
    T[i,j] = \int (x-A)^i e^(-a(x-A)^2) (d^2/dx^2) [(x-B)^j e^(-b(x-B)^2)] dx

    where the integral is over all space.

    We use the formula:
    T[i, j] = j(j-1)S[i,j-2] -  2b(2j + 1)S[i,j] + 4b**2 S[i,j+2]
    """
    if S.shape != (g1.max_degree + 1, g2.max_degree + 1):
        raise ValueError(
            f"Expected S.shape to be {(g1.max_degree + 1, g2.max_degree + 1)}, got {S.shape}"
        )

    T = np.zeros((S.shape[0], S.shape[1] - 2), dtype=S.dtype)

    b = g2.exponent
    j_array = np.arange(2, T.shape[1])

    T[:, 2:] += (j_array[:, 2:] * (j_array[:, 2:] - 1)) * S[:, :-4]
    T[:, :] -= 2 * b * (2 * j_array + 1) * S[:, :-2]
    T[:, :] += 4 * b**2 * S[:, 2:]

    return T
