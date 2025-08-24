import numpy as np

from integrals import gaussian


def _overlap_1d_base_case(
    S: np.ndarray,
    g1: gaussian.CartesianGaussian1d,
    g2: gaussian.CartesianGaussian1d,
) -> None:
    p = g1.exponent + g2.exponent
    K = gaussian.overlap_prefactor_1d(g1, g2)
    S[0, 0] = np.sqrt(np.pi / p) * K


def _overlap_1d_vertical_transfer(
    S: np.ndarray,
    g1: gaussian.CartesianGaussian1d,
    g2: gaussian.CartesianGaussian1d,
) -> None:
    p = g1.exponent + g2.exponent
    P = (g1.exponent * g1.center + g2.exponent * g2.center) / p

    for i in range(1, g1.max_degree + g2.max_degree + 1):
        S[i, 0] = (P - g1.center) * S[i - 1, 0] + ((i - 1) / (2 * p)) * S[
            i - 2, 0
        ]


def _overlap_1d_horizontal_transfer(
    S: np.ndarray,
    g1: gaussian.CartesianGaussian1d,
    g2: gaussian.CartesianGaussian1d,
) -> None:
    diff = g1.center - g2.center

    for j in range(1, g2.max_degree + 1):
        for i in range(0, g1.max_degree + g2.max_degree - j + 1):
            S[i, j] = diff * S[i, j - 1] + S[i + 1, j - 1]


def overlap_1d(
    g1: gaussian.CartesianGaussian1d, g2: gaussian.CartesianGaussian1d
) -> np.ndarray:
    """
    The output S has shape (g1.max_degree+1, g2.max_degree+1).
    For each 0 <= i <= g1.max_degree+1 and 0 <= j <= g2.max_degree+1,
    S[i,j] = S_{ij}(g1.exponent, g2.exponent, g1.center, g2.center)
    """
    S = np.zeros((g1.max_degree + g2.max_degree + 1, g2.max_degree + 1))
    _overlap_1d_base_case(S, g1, g2)
    print("Base case overlap matrix:")
    print(S)

    _overlap_1d_vertical_transfer(S, g1, g2)
    print("After vertical transfer overlap matrix:")
    print(S)

    _overlap_1d_horizontal_transfer(S, g1, g2)
    print("After horizontal transfer overlap matrix:")
    print(S)
    return S[:g1.max_degree + 1, :]
