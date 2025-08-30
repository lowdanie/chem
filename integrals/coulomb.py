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
    P: np.ndarray,
    C: np.ndarray,
) -> None:
    K = gaussian.overlap_prefactor_3d(g1, g2)
    dist_sq = np.sum(np.square(g1.center - g2.center))

    for n in range(V.shape[3]):
        V[0, 0, 0, n] = K * boys(n, p * dist_sq)


def _V_vertical_transfer_base(
    V: np.ndarray, dim: int, s: float, p: float, A: float, C: float, P: float
) -> np.ndarray:
    n_size = V.shape[3]
    d_size = V.shape[dim]

    return (P - A) * V[..., : n_size - 1] - (s / p) * (P - C) * V[..., 1:n_size]


def _V_vertical_transfer(
    V: np.ndarray, dim: int, s: float, p: float, A: float, C: float, P: float
) -> np.ndarray:
    pass


def _V_vertical_transfer_x(
    V: np.ndarray, s: float, p: float, A: float, C: float, P: float
) -> None:
    V_size = V.shape[0]

    # fmt: off
    V[1, 0, 0, :V_size-1] = (
        (P - A) * V[0, 0, 0, :V_size-1]
        - (s/p)*(P - C) * V[0, 0, 0, 1:V_size]
    )
    # fmt: on

    for i in range(2, V_size):
        V[i, 0, 0, : V_size - i] = (
            (P - A) * V[i - 1, 0, 0, : V_size - i]
            - (s / p) * (P - C) * V[i - 1, 0, 0, 1 : V_size - i + 1]
            + ((i - 1) / (2 * p)) * V[i - 2, 0, 0, : V_size - i]
            - (((i - 1) * s) / (2 * p**2)) * V[i - 2, 0, 0, 1 : V_size - i + 1]
        )


def _V_vertical_transfer_y(
    V: np.ndarray, s: float, p: float, A: float, C: float, P: float
) -> None:
    V_size = V.shape[1]

    # fmt: off
    V[:, 1, 0, :V_size-1] = (
        (P - A) * V[:, 0, 0, :V_size-1]
        - (s/p)*(P - C) * V[:, 0, 0, 1:V_size]
    )
    # fmt: on

    for j in range(2, V_size):
        V[:, j, 0, : V_size - j] = (
            (P - A) * V[:, j - 1, 0, : V_size - j]
            - (s / p) * (P - C) * V[:, j - 1, 0, 1 : V_size - j + 1]
            + ((j - 1) / (2 * p)) * V[:, j - 2, 0, : V_size - j]
            - (((j - 1) * s) / (2 * p**2)) * V[:, j - 2, 0, 1 : V_size - j + 1]
        )


def _V_vertical_transfer_z(
    V: np.ndarray, s: float, p: float, A: float, C: float, P: float
) -> None:
    V_size = V.shape[2]

    # fmt: off
    V[:, :, 1, :V_size-1] = (
        (P - A) * V[:, :, 0, :V_size-1]
        - (s/p)*(P - C) * V[:, :, 0, 1:V_size]
    )
    # fmt: on

    for k in range(2, V_size):
        V[:, :, k, : V_size - k] = (
            (P - A) * V[:, :, k - 1, : V_size - k]
            - (s / p) * (P - C) * V[:, :, k - 1, 1 : V_size - k + 1]
            + ((k - 1) / (2 * p)) * V[:, :, k - 2, : V_size - k]
            - (((k - 1) * s) / (2 * p**2)) * V[:, :, k - 2, 1 : V_size - k + 1]
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
    _V_base_case(V, g1, g2, p, P, C)
    _V_vertical_transfer_x(V, s, p, A[0], C[0], P[0])
    _V_vertical_transfer_y(V, s, p, A[1], C[1], P[1])
    _V_vertical_transfer_z(V, s, p, A[2], C[2], P[2])

    return V[:, :, :, 0]
