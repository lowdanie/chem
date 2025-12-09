import numpy as np

from integrals import gaussian
from integrals import overlap


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
    T[i,j] = integral (x-A)^i e^(-a(x-A)^2) (d^2/dx^2) [(x-B)^j e^(-b(x-B)^2)] dx

    where the integral is over all space.

    We use the formula:
    T[i, j] = j(j-1)S[i,j-2] -  2b(2j + 1)S[i,j] + 4b**2 S[i,j+2]
    """
    if S.shape != (g1.max_degree + 1, g2.max_degree + 1):
        raise ValueError(
            f"Expected S.shape to be {(g1.max_degree + 1, g2.max_degree + 1)}, got {S.shape}"
        )

    if g2.max_degree < 2:
        return np.empty((S.shape[0], 0), dtype=S.dtype)

    T = np.zeros((S.shape[0], S.shape[1] - 2), dtype=S.dtype)

    b = g2.exponent
    j_array = np.arange(T.shape[1])[np.newaxis, :]

    T[:, 2:] += (j_array[:, 2:] * (j_array[:, 2:] - 1)) * S[:, :-4]
    T[:, :] -= 2 * b * (2 * j_array + 1) * S[:, :-2]
    T[:, :] += 4 * b**2 * S[:, 2:]

    return T


def kinetic_3d_from_overlap_1d(
    S_x: np.ndarray,
    S_y: np.ndarray,
    S_z: np.ndarray,
    g1: gaussian.GaussianBasis3d,
    g2: gaussian.GaussianBasis3d,
) -> np.ndarray:
    """
    We assume that:
    S_x = overlap_1d(g1_x, g2_x)
    S_y = overlap_1d(g1_y, g2_y)
    S_z = overlap_1d(g1_z, g2_z)

    where g1_x, g1_y, g1_z are the 1d Gaussian basis functions corresponding to
    g1 and similarly for g2.

    Set
    d1 = g1.max_degree, d2 = g2.max_degree
    a = g1.exponent, b = g2.exponent
    A = g1.center, B = g2.center

    Then the output T has shape
    (d1 + 1, d1 + 1, d1 + 1, d2 - 1, d2 - 1, d2 - 1)

    And is defined by:

    T[ix, iy, iz, jx, jy, jz] =
        integral integral integral
            (x-Ax)^ix (y-Ay)^iy (z-Az)^iz e^(-a((x-Ax)^2+(y-Ay)^2+(z-Az)^2))
            (d^2/dx^2 + d^2/dy^2 + d^2/dz^2)
            [(x-Bx)^jx (y-By)^jy (z-Bz)^jz e^(-b((x-Bx)^2+(y-By)^2+(z-Bz)^2))]
        dx dy dz

    where the integral is over all space.

    We use the formula:
    T[ix, iy, iz, jx, jy, jz] =
        T_x[ix, jx] * S_y[iy, jy] * S_z[iz, jz] +
        S_x[ix, jx] * T_y[iy, jy] * S_z[iz, jz] +
        S_x[ix, jx] * S_y[iy, jy] * T_z[iz, jz]
    """
    T_x = kinetic_1d_from_overlap_1d(
        S_x,
        gaussian.gaussian_3d_to_1d(g1, 0),
        gaussian.gaussian_3d_to_1d(g2, 0),
    )
    T_y = kinetic_1d_from_overlap_1d(
        S_y,
        gaussian.gaussian_3d_to_1d(g1, 1),
        gaussian.gaussian_3d_to_1d(g2, 1),
    )
    T_z = kinetic_1d_from_overlap_1d(
        S_z,
        gaussian.gaussian_3d_to_1d(g1, 2),
        gaussian.gaussian_3d_to_1d(g2, 2),
    )

    d1 = g1.max_degree
    d2 = g2.max_degree

    T = np.zeros((d1 + 1, d1 + 1, d1 + 1, d2 - 1, d2 - 1, d2 - 1))

    T += np.einsum("ad,be,cf->abcdef", T_x, S_y[:, :-2], S_z[:, :-2])
    T += np.einsum("ad,be,cf->abcdef", S_x[:, :-2], T_y, S_z[:, :-2])
    T += np.einsum("ad,be,cf->abcdef", S_x[:, :-2], S_y[:, :-2], T_z)

    return T


def kinetic_3d(
    g1: gaussian.GaussianBasis3d, g2: gaussian.GaussianBasis3d
) -> np.ndarray:
    """Computes the 3D Kinetic Energy matrix.

    This acts as a wrapper around the `kinetic_3d_from_overlap_1d` kernel.
    It bumps up the g2 basis by +2 angular momentum so that
    the output has the correct shape (d2 + 1).

    The output is an an array with shape
    (d1+1, d1+1, d1+1, d2+1, d2+1, d2+1)
    """

    # Increase the degree of g2 by +2
    # We need this because the kinetic operator involves 2nd derivatives,
    # so we need overlaps of higher angular momentum to resolve it.
    g2_boosted = gaussian.GaussianBasis3d(
        max_degree=g2.max_degree + 2, exponent=g2.exponent, center=g2.center
    )

    # Compute 1D Overlaps using the boosted g2
    # Shape: (d1+1, d2+3)
    g1x, g1y, g1z = [gaussian.gaussian_3d_to_1d(g1, i) for i in range(3)]
    g2x, g2y, g2z = [
        gaussian.gaussian_3d_to_1d(g2_boosted, i) for i in range(3)
    ]

    S_x = overlap.overlap_1d(g1x, g2x)
    S_y = overlap.overlap_1d(g1y, g2y)
    S_z = overlap.overlap_1d(g1z, g2z)

    return kinetic_3d_from_overlap_1d(S_x, S_y, S_z, g1, g2_boosted)
