import dataclasses
import pytest

from jax import jit
import numpy as np

import slaterform as sf


@dataclasses.dataclass
class _TestCase1d:
    g1: sf.GaussianBasis1d
    g2: sf.GaussianBasis1d
    expected: np.ndarray


@dataclasses.dataclass
class _TestCase3dSparse:
    g1: sf.GaussianBasis3d
    g2: sf.GaussianBasis3d
    expected_shape: tuple[int, ...]
    expected_values: list[tuple[tuple[int, ...], float]]


# The expected values were computed using sympy.
# import sympy as sp
# import numpy as np
#
# def build_cartesian_gaussian_1d(x, i, a, A):
#   return (x - A)**i * sp.exp(-a * (x - A)**2)
#
# def overlap_1d(i, j, a, b, A, B):
#   x = sp.symbols('x')
#   g1 = build_cartesian_gaussian_1d(x, i, a, A)
#   g2 = build_cartesian_gaussian_1d(x, j, b, B)
#   integrand = g1 * g2
#   res = sp.integrate(integrand, (x, -sp.oo, sp.oo))
#   return float(res.evalf())
#
# def overlap_matrix_1d(d1, d2, a, b, A, B):
#   S = np.zeros((d1+1, d2+1), dtype=np.float64)
#   for i in range(d1+1):
#     for j in range(d2+1):
#       print(f'{i}, {j}')
#       S[i, j] = overlap_1d(i, j, a, b, A, B)
#   return S
#
# d1 = g1.max_degree
# d2 = g2.max_degree
# a = sp.Rational(1, 2)
# b = sp.Rational(1, 5)
# A = sp.Integer(-2)
# B = sp.Integer(1)
#
# S = overlap_matrix_1d(g1.max_degree, g2.max_degree, a, b, A, B)
_TEST_CASES_1D = [
    _TestCase1d(
        g1=sf.GaussianBasis1d(max_degree=2, exponent=0.5, center=-2.0),
        g2=sf.GaussianBasis1d(max_degree=3, exponent=0.2, center=1.0),
        expected=np.array(
            [
                [
                    0.5856623378112075,
                    -1.2549907238811588,
                    3.1075960781819174,
                    -8.4519783445057648,
                ],
                [
                    0.5019962895524636,
                    -0.6573760934615595,
                    0.8708098900399878,
                    -0.5854184134722608,
                ],
                [
                    0.8486127751958312,
                    -1.1013183903446904,
                    2.0270112566477025,
                    -4.6728934075375097,
                ],
            ]
        ),
    ),
    _TestCase1d(
        g1=sf.GaussianBasis1d(max_degree=0, exponent=0.5, center=-2.0),
        g2=sf.GaussianBasis1d(max_degree=0, exponent=0.2, center=1.0),
        expected=np.array([[0.5856623378112075]]),
    ),
    _TestCase1d(
        g1=sf.GaussianBasis1d(max_degree=0, exponent=0.5, center=-2.0),
        g2=sf.GaussianBasis1d(max_degree=2, exponent=0.2, center=1.0),
        expected=np.array(
            [[0.5856623378112075, -1.2549907238811588, 3.1075960781819174]]
        ),
    ),
    _TestCase1d(
        g1=sf.GaussianBasis1d(max_degree=2, exponent=0.5, center=-2.0),
        g2=sf.GaussianBasis1d(max_degree=0, exponent=0.2, center=1.0),
        expected=np.array(
            [
                [0.5856623378112075],
                [0.5019962895524636],
                [0.8486127751958312],
            ]
        ),
    ),
]

# The expected values were computed using sympy.
# def build_cartesian_gaussian_3d(coords, powers, exponent, center):
#     x, y, z = coords
#     cx, cy, cz = center
#     ix, iy, iz = powers

#     monom = (x - cx) ** ix * (y - cy) ** iy * (z - cz) ** iz
#     dist_sq = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
#     return monom * sp.exp(-exponent * dist_sq)
#
# def overlap_3d(powers1, powers2, a, b, A, B):
#   coords = sp.symbols('x,y,z')
#   x, y, z = coords
#
#   g1 = build_cartesian_gaussian_3d(coords, powers1, a, A)
#   g2 = build_cartesian_gaussian_3d(coords, powers2, b, B)
#   integrand = g1 * g2
#   res = sp.integrate(integrand, (x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo), (z, -sp.oo, sp.oo))
#   return float(res.evalf())
#
# a = sp.Rational(1, 2)
# b = sp.Rational(1, 5)
# A = sp.Integer(-2)
# B = sp.Integer(1)
#
# # The expected value of (ix, iy, iz, jx, jy, jz) is:
# expected = overlap_3d((ix, iy, iz), (jx, jy, jz), a, b, A, B)
_TEST_CASES_3D = [
    _TestCase3dSparse(
        g1=sf.GaussianBasis3d(
            max_degree=2,
            exponent=0.5,
            center=np.array([-2.0, 0.0, 1.0]),
        ),
        g2=sf.GaussianBasis3d(
            max_degree=3,
            exponent=0.2,
            center=np.array([1.0, 2.0, -1.0]),
        ),
        expected_shape=(3, 3, 3, 4, 4, 4),
        expected_values=[
            ((2, 1, 0, 1, 0, 1), -1.286743800702015),
            ((0, 1, 2, 2, 1, 0), -0.4723760464371185),
            ((0, 0, 0, 0, 0, 0), 0.8382288007131306),
            ((1, 1, 1, 1, 1, 1), -0.00979663746381656),
            ((2, 2, 2, 2, 2, 2), 7.020504702738119),
            ((2, 2, 2, 3, 3, 3), 98.95957347011944),
            ((0, 0, 0, 3, 3, 3), 432.10882854250946),
            ((2, 2, 2, 0, 0, 0), 1.3157489724221587),
        ],
    ),
    _TestCase3dSparse(
        g1=sf.GaussianBasis3d(
            max_degree=0,
            exponent=0.5,
            center=np.array([-2.0, 0.0, 1.0]),
        ),
        g2=sf.GaussianBasis3d(
            max_degree=3,
            exponent=0.2,
            center=np.array([1.0, 2.0, -1.0]),
        ),
        expected_shape=(1, 1, 1, 4, 4, 4),
        expected_values=[
            ((0, 0, 0, 0, 0, 0), 0.8382288007131306),
            ((0, 0, 0, 3, 3, 3), 432.10882854250946),
        ],
    ),
    _TestCase3dSparse(
        g1=sf.GaussianBasis3d(
            max_degree=2,
            exponent=0.5,
            center=np.array([-2.0, 0.0, 1.0]),
        ),
        g2=sf.GaussianBasis3d(
            max_degree=0,
            exponent=0.2,
            center=np.array([1.0, 2.0, -1.0]),
        ),
        expected_shape=(3, 3, 3, 1, 1, 1),
        expected_values=[
            ((0, 0, 0, 0, 0, 0), 0.8382288007131306),
            ((2, 2, 2, 0, 0, 0), 1.3157489724221587),
        ],
    ),
]


@pytest.mark.parametrize("case", _TEST_CASES_1D)
def test_overlap_1d(case):
    S = jit(sf.integrals.overlap_1d)(case.g1, case.g2)

    np.testing.assert_allclose(S, case.expected, rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("case", _TEST_CASES_3D)
def test_overlap_3d(case):
    S = jit(sf.integrals.overlap_3d)(case.g1, case.g2)
    assert S.shape == case.expected_shape

    for coords, expected in case.expected_values:
        actual = S[coords]
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-15)
