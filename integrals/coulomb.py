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
    V: np.ndarray, size_n: int, s: float, p: float, A: float, C: float, P: float
) -> None:
    """
    V has 2 <= n_dims <= 4 dimensions with shape (d,..,d,3*d).

    We apply the following recursive formula to the last 2 dimensions:

    V[...,i,n] =        (P - A)V[...,i-1,n]
                    -(s/p)(P-C)V[...,i-1,n+1]
                   +((i-1)/(2p)V[...,i-2,n]
             -(((i-1)s)/(2p^2))V[...,i-2,n+1]

    for all 0 <= n < size_n - i

    We assume that V has already been computed for V[...,0,:size_n].
    """
    # fmt: off
    V[..., 1, : size_n - 1] = (
        (P - A) * V[..., 0, : size_n - 1]
        - (s/p)*(P - C) * V[..., 0, 1 : size_n]
    )
    # fmt: on

    for i in range(2, V.shape[-2]):
        V[..., i, : size_n - i] = (
            (P - A) * V[..., i - 1, : size_n - i]
            - (s / p) * (P - C) * V[..., i - 1, 1 : size_n - i + 1]
            + ((i - 1) / (2 * p)) * V[..., i - 2, : size_n - i]
            - (((i - 1) * s) / (2 * p**2)) * V[..., i - 2, 1 : size_n - i + 1]
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

    # The last dimension of V needs to be 3 * size_d to have enough space
    # for 3 vertical transfers.
    size_d = g1.max_degree + 1
    V = np.zeros((size_d, size_d, size_d, 3 * size_d), dtype=np.float64)

    _V_base_case(V, g1, g2, p, C, P)
    _V_vertical_transfer(V[:, 0, 0, :], 3 * size_d, s, p, A[0], C[0], P[0])
    _V_vertical_transfer(V[:, :, 0, :], 2 * size_d, s, p, A[1], C[1], P[1])
    _V_vertical_transfer(V, size_d, s, p, A[2], C[2], P[2])

    return V[:, :, :, 0]


def _one_electron_horizontal_transfer(
    I: np.ndarray, A: float, B: float
) -> None:
    """
    I has 4 <= d <= 6 dimensions.

    We apply the following recursive formula to dimensions d and d-3:

    I[..., i, :, :, j] = (A - B)I[..., i, :, :, j-1] + I[..., i+1, :, :, j-1]

    for all 0 <= i < I.shape[d-3] - j and 1 <= j < I.shape[d].
    """
    dim_1 = len(I.shape) - 4
    dim_2 = len(I.shape) - 1

    size_1 = I.shape[dim_1]
    size_2 = I.shape[dim_2]

    for j in range(1, size_2):
        # fmt: off
        I[...,:size_1 - j, :, : , j] = (
            (A - B) * I[..., :size_1 - j, :, :, j - 1]
            + I[..., 1 : size_1 - j + 1, :, : , j - 1]
        )
        # fmt: on


def one_electron(
    g1: gaussian.GaussianBasis3d,
    g2: gaussian.GaussianBasis3d,
    C: np.ndarray,
) -> np.ndarray:
    """
    Compute the one electron Coulomb integral I defined by:
    G1(x,y,z) = (x-Ax)^ix (y-Ay)^iy (z-Az)^iz e^(-a((x-Ax)^2+(y-Ay)^2+(z-Az)^2))
    G2(x,y,z) = (x-Bx)^jx (y-By)^jy (z-Bz)^jz e^(-b((x-Bx)^2+(y-By)^2+(z-Bz)^2))

    I[ix,iy,iz,jx,jy,jz] =
        integral
            G1(x,y,z) * G2(x,y,z) / sqrt((x-Cx)^2+(y-Cy)^2+(z-Cz)^2))
        dx dy dz

    where:
    g1 = GaussianBasis3d(max_degree=d1, exponent=a, center=A)
    g2 = GaussianBasis3d(max_degree=d2, exponent=b, center=B)

    The output has shape:
    (d1+1, d1+1, d1+1, d2+1, d2+1, d2+1)
    """
    a, A = g1.exponent, g1.center
    b, B = g2.exponent, g2.center

    # Pad g1 so that we have enough space to do horizontal transfers.
    padded_g1 = gaussian.GaussianBasis3d(
        max_degree=g1.max_degree + g2.max_degree,
        exponent=g1.exponent,
        center=g1.center,
    )

    I = np.zeros(
        (padded_g1.max_degree + 1,) * 3 + (g2.max_degree + 1,) * 3,
        dtype=np.float64,
    )

    I[..., 0, 0, 0] = (2 * np.pi / (a + b)) * _V(padded_g1, g2, a + b, C)

    _one_electron_horizontal_transfer(I[..., 0, 0], A[0], B[0])
    _one_electron_horizontal_transfer(I[..., 0], A[1], B[1])
    _one_electron_horizontal_transfer(I, A[2], B[2])

    return I[: g1.max_degree + 1, : g1.max_degree + 1, : g1.max_degree + 1, ...]


def two_electron(
    g1: gaussian.GaussianBasis3d,
    g2: gaussian.GaussianBasis3d,
    g3: gaussian.GaussianBasis3d,
    g4: gaussian.GaussianBasis3d,
) -> float:
    """
    Compute the two electron Coulomb integral:

    G1(x,y,z) = (x-Ax)^ix (y-Ay)^iy (z-Az)^iz e^(-a((x-Ax)^2+(y-Ay)^2+(z-Az)^2))
    G2(x,y,z) = (x-Bx)^jx (y-By)^jy (z-Bz)^jz e^(-b((x-Bx)^2+(y-By)^2+(z-Bz)^2))
    G3(x,y,z) = (x-Cx)^kx (y-Cy)^ky (z-Cz)^kz e^(-c((x-Cx)^2+(y-Cy)^2+(z-Cz)^2))
    G4(x,y,z) = (x-Dx)^lx (y-Dy)^ly (z-Dz)^lz e^(-d((x-Dx)^2+(y-Dy)^2+(z-Dz)^2))

    I[ix,iy,iz,jx,jy,jz,kx,ky,kz,lx,ly,lz] =
        integral
            G1(x1,y1,z1) * G2(x1,y1,z1) *
            G3(x2,y2,z2) * G4(x2,y2,z2) /
            sqrt((x1-x2)^2+(y1-y2)^2+(z1-z2)^2)
        dx1 dy1 dz1 dx2 dy2 dz2

    where:
    g1 = GaussianBasis3d(max_degree=d1, exponent=a, center=A)
    g2 = GaussianBasis3d(max_degree=d2, exponent=b, center=B)
    g3 = GaussianBasis3d(max_degree=d3, exponent=c, center=C)
    g4 = GaussianBasis3d(max_degree=d4, exponent=d, center=D)

    The output is an array with shape:
    (d1+1, d1+1, d1+1,
     d2+1, d2+1, d2+1,
     d3+1, d3+1, d3+1,
     d4+1, d4+1, d4+1)
    """
    # Pad g3 so that we have enough space to do horizontal transfers
    # to g4.
    padded_g3 = gaussian.GaussianBasis3d(
        max_degree=g3.max_degree + g4.max_degree,
        exponent=g3.exponent,
        center=g3.center,
    )

    # Pad g1 so that we have enough space to do horizontal transfers
    # to g2 and electron transfers to padded_g3.
    padded_g1 = gaussian.GaussianBasis3d(
        max_degree=g1.max_degree + g2.max_degree + padded_g3.max_degree,
        exponent=g1.exponent,
        center=g1.center,
    )

    I = np.zeros(
        (padded_g1.max_degree + 1,) * 3
        + (g2.max_degree + 1,) * 3
        + (padded_g3.max_degree + 1,) * 3
        + (g4.max_degree + 1,) * 3,
        dtype=np.float64,
    )

    _two_electron_base_case(I, padded_g1, g2, padded_g3, g4)

    A, B = g1.center, g2.center
    I_ab = I[..., 0, 0, 0, 0, 0, 0]
    _two_electron_horizontal_transfer(I_ab[..., 0, 0], A[0], B[0])
    _two_electron_horizontal_transfer(I_ab[..., 0], A[1], B[1])
    _two_electron_horizontal_transfer(I_ab, A[2], B[2])
