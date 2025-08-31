import numpy as np
from scipy import special
from integrals import gaussian


def boys(n, x):
    """
    Compute the Boys function:
    F_n(x) = integral_0^1 t^(2n) exp(-x t^2) dt

    We use the fact that the Boys function can be expressed in terms of the
    confluent hyper-geometric function:
    """
    return special.hyp1f1(n + 0.5, n + 1.5, -x) / (2.0 * n + 1.0)


def _V_base_case(
    V: np.ndarray,
    g1: gaussian.GaussianBasis3d,
    g2: gaussian.GaussianBasis3d,
    p: float,
    C: np.ndarray,
    P: np.ndarray,
) -> None:
    K = gaussian.overlap_prefactor_3d(g1, g2)
    dist_sq = np.sum(np.square(P - C))

    for n in range(V.shape[3]):
        V[0, 0, 0, n] = K * boys(n, p * dist_sq)


def _V_vertical_transfer(
    V: np.ndarray, s: float, p: float, A: float, C: float, P: float
) -> None:
    V_size = V.shape[0]

    # fmt: off
    V[..., 1, :V_size-1] = (
        (P - A) * V[..., 0, :V_size - 1]
        - (s/p)*(P - C) * V[..., 0, 1 : V_size]
    )
    # fmt: on

    for i in range(2, V_size):
        V[..., i, : V_size - i] = (
            (P - A) * V[..., i - 1, : V_size - i]
            - (s / p) * (P - C) * V[..., i - 1, 1 : V_size - i + 1]
            + ((i - 1) / (2 * p)) * V[..., i - 2, : V_size - i]
            - (((i - 1) * s) / (2 * p**2)) * V[..., i - 2, 1 : V_size - i + 1]
        )


def _V(
    g1: gaussian.GaussianBasis3d,
    g2: gaussian.GaussianBasis3d,
    s: float,
    C: np.ndarray,
) -> np.ndarray:
    """Auxiliary function for computing one and two electron Coulomb integrals."""
    p = g1.exponent + g2.exponent
    P = (g1.exponent * g1.center + g2.exponent * g2.center) / p
    A = g1.center

    V_size = g1.max_degree + 1
    V = np.zeros((V_size, V_size, V_size, V_size), dtype=np.float64)
    _V_base_case(V, g1, g2, p, C, P)
    _V_vertical_transfer(V[:, 0, 0, :], s, p, A[0], C[0], P[0])
    _V_vertical_transfer(V[:, :, 0, :], s, p, A[1], C[1], P[1])
    _V_vertical_transfer(V, s, p, A[2], C[2], P[2])

    return V[:, :, :, 0]


def _one_electron_horizontal_transfer(
    I: np.ndarray, A: float, B: float
) -> None:
    dim_1 = len(I.shape) - 3
    dim_2 = len(I.shape) - 1

    size_1 = I.shape[dim_1]
    size_2 = I.shape[dim_2]

    for j in range(1, size_2):
        # fmt: off
        I[:size_1 - j, ..., j] = (
            (A - B) * I[:size_1 - j, ..., j - 1]
            + I[1 : size_1 - j + 1, ..., j - 1]
        )
        # fmt: on


def _one_electron(
    g1: gaussian.GaussianBasis3d,
    g2: gaussian.GaussianBasis3d,
    C: np.ndarray,
) -> np.ndarray:
    """
    Compute the one electron Coulomb integral:
    I[ix,iy,iz,jx,jy,jz] =
        integral (x-Ax)^ix (y-Ay)^iy (z-Az)^iz e^(-a((x-Ax)^2+(y-Ay)^2+(z-Az)^2))
                 (x-Bx)^jx (y-By)^jy (z-Bz)^jz e^(-b((x-Bx)^2+(y-By)^2+(z-Bz)^2))
                 (1/||r-C||)
        dx dy dz

    where:
    g1 = GaussianBasis3d(max_degree=d1, exponent=a, center=A)
    g2 = GaussianBasis3d(max_degree=d2, exponent=b, center=B)

    The output has shape:
    (d1+1, d1+1, d1+1, d2+1, d2+1, d2+1)
    """
    d1, a, A = g1.max_degree, g1.exponent, g1.center
    d2, b, B = g2.max_degree, g2.exponent, g2.center

    I = np.zeros(
        (d1 + d2 + 1, d1 + d2 + 1, d1 + d2 + 1, d2 + 1, d2 + 1, d2 + 1)
    )

    I[..., 0, 0, 0] = (2 * np.pi / (a + b)) * _V(g1, g2, a + b, C)
    _one_electron_horizontal_transfer(I[..., 0, 0], A[0], B[0])
    _one_electron_horizontal_transfer(I[..., 0], A[1], B[1])
    _one_electron_horizontal_transfer(I, A[2], B[2])

    return I[: d1 + 1, : d1 + 1, : d1 + 1, ...]
