import numpy as np

from slaterform.integrals import gaussian


def _overlap_1d_base_case(
    S: np.ndarray,
    g1: gaussian.GaussianBasis1d,
    g2: gaussian.GaussianBasis1d,
) -> None:
    p = g1.exponent + g2.exponent
    K = gaussian.overlap_prefactor_1d(g1, g2)
    S[0, 0] = np.sqrt(np.pi / p) * K


def _overlap_1d_vertical_transfer(
    S: np.ndarray,
    g1: gaussian.GaussianBasis1d,
    g2: gaussian.GaussianBasis1d,
) -> None:
    p = g1.exponent + g2.exponent
    P = (g1.exponent * g1.center + g2.exponent * g2.center) / p

    for i in range(1, g1.max_degree + g2.max_degree + 1):
        # fmt: off
        S[i, 0] = (
            (P - g1.center) * S[i - 1, 0] 
            + ((i - 1) / (2 * p)) * S[i - 2, 0]
        )
        # fmt: on


def _overlap_1d_horizontal_transfer(
    S: np.ndarray,
    g1: gaussian.GaussianBasis1d,
    g2: gaussian.GaussianBasis1d,
) -> None:
    diff = g1.center - g2.center
    size_1 = g1.max_degree + g2.max_degree + 1
    size_2 = g2.max_degree + 1

    for j in range(1, size_2):
        S[: size_1 - j, j] = (
            diff * S[: size_1 - j, j - 1] + S[1 : size_1 - j + 1, j - 1]
        )


def overlap_1d(
    g1: gaussian.GaussianBasis1d, g2: gaussian.GaussianBasis1d
) -> np.ndarray:
    """
    Set:
    d1 = g1.max_degree
    a = g1.exponent
    A = g1.center
    d2 = g2.max_degree
    b = g2.exponent
    B = g2.center

    The output S has shape (d1+1, d2+1).

    For each 0 <= i <= d1+1 and 0 <= j <= d2+1:
    S[i,j] = integral (x-A)^i (x-B)^j e^(-a(x-A)^2) e^(-b(x-B)^2) dx

    where the integral is over all space.
    """
    S = np.zeros((g1.max_degree + g2.max_degree + 1, g2.max_degree + 1))
    _overlap_1d_base_case(S, g1, g2)
    _overlap_1d_vertical_transfer(S, g1, g2)
    _overlap_1d_horizontal_transfer(S, g1, g2)
    return S[: g1.max_degree + 1, :]


def overlap_3d_from_1d(
    S_x: np.ndarray, S_y: np.ndarray, S_z: np.ndarray
) -> np.ndarray:
    """
    Given three overlap matrices S_x, S_y, and S_z, compute the full overlap
    matrix S of shape
    (S_x.shape[0], S_y.shape[0], S_z.shape[0],
     S_x.shape[1], S_y.shape[1], S_z.shape[1])

    defined by:
    S[ix, iy, iz, jx, jy, jz] = S_x[ix, jx] * S_y[iy, jy] * S_z[iz, jz]

    """
    return np.einsum("ad,be,cf->abcdef", S_x, S_y, S_z)


def overlap_3d(
    g1: gaussian.GaussianBasis3d, g2: gaussian.GaussianBasis3d
) -> np.ndarray:
    """
    If
    g1.max_degree = d1
    g1.exponent = a
    g1.center = (Ax, Ay, Az)

    g2.max_degree = d2
    g2.exponent = b
    g2.center = (Bx, By, Bz)

    Then the output S has shape
    (d1+1, d1+1, d1+1, d2+1, d2+1, d2+1).

    And for all
    (ix, iy, iz) with 0 <= ix, iy, iz <= d1 and
    (jx, jy, jz) with 0 <= jx, jy, jz <= d2:

    S[ix, iy, iz, jx, jy, jz] =
    integral integral integral
        (x-Ax)^ix (y-Ay)^iy (z-Az)^iz e^(-a((x-Ax)^2+(y-Ay)^2+(z-Az)^2))
        (x-Bx)^jx (y-By)^jy (z-Bz)^jz e^(-b((x-Bx)^2+(y-By)^2+(z-Bz)^2))
    dx dy dz

    where the integral is over all space.
    """
    S_x = overlap_1d(
        gaussian.gaussian_3d_to_1d(g1, 0), gaussian.gaussian_3d_to_1d(g2, 0)
    )
    S_y = overlap_1d(
        gaussian.gaussian_3d_to_1d(g1, 1), gaussian.gaussian_3d_to_1d(g2, 1)
    )
    S_z = overlap_1d(
        gaussian.gaussian_3d_to_1d(g1, 2), gaussian.gaussian_3d_to_1d(g2, 2)
    )
    return overlap_3d_from_1d(S_x, S_y, S_z)
