import dataclasses
import pytest

from jax import jit
import numpy as np

import slaterform as sf


@dataclasses.dataclass
class _TestCase:
    g1: sf.GaussianBasis3d
    g2: sf.GaussianBasis3d
    expected_shape: tuple[int, int, int, int, int, int]
    expected_values: list[tuple[tuple[int, ...], float]]


# The expected values were computed using sympy:
# import sympy as sp
#
# def build_cartesian_gaussian_3d(coords, powers, exponent, center):
#     x, y, z = coords
#     cx, cy, cz = center
#     ix, iy, iz = powers
#
#     monom = (x - cx) ** ix * (y - cy) ** iy * (z - cz) ** iz
#     dist_sq = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
#     return monom * sp.exp(-exponent * dist_sq)
#
# def kinetic_3d(powers1, powers2, a, b, A, B):
#   coords = sp.symbols('x,y,z')
#   x, y, z = coords
#
#   g1 = build_cartesian_gaussian_3d(coords, powers1, a, A)
#   g2 = build_cartesian_gaussian_3d(coords, powers2, b, B)
#   nabla_g2 = sp.diff(g2, x, 2) + sp.diff(g2, y, 2) + sp.diff(g2, z, 2)
#   integrand = g1 * nabla_g2
#
#   res = sp.integrate(integrand, (x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo), (z, -sp.oo, sp.oo))
#   return float(res.evalf())
#
# a = sp.Rational(1, 2)  # 0.5
# A = [sp.Integer(-2), sp.Integer(0), sp.Integer(1)]
# b = sp.Rational(1, 5)  # 0.2
# B = [sp.Integer(1), sp.Integer(2), sp.Integer(-1)]
#
# expected = kinetic_3d(powers[:3], powers[3:], a, b, A, B)
_TEST_CASES = [
    _TestCase(
        g1=sf.GaussianBasis3d(
            max_degree=0,
            exponent=0.5,
            center=np.array([-2.0, 0.0, 1.0]),
        ),
        g2=sf.GaussianBasis3d(
            max_degree=0,
            exponent=0.2,
            center=np.array([1.0, 2.0, -1.0]),
        ),
        expected_shape=(1, 1, 1, 1, 1, 1),
        expected_values=[
            ((0, 0, 0, 0, 0, 0), 0.4447744656845183),
        ],
    ),
    _TestCase(
        g1=sf.GaussianBasis3d(
            max_degree=1,
            exponent=0.5,
            center=np.array([-2.0, 0.0, 1.0]),
        ),
        g2=sf.GaussianBasis3d(
            max_degree=0,
            exponent=0.2,
            center=np.array([1.0, 2.0, -1.0]),
        ),
        expected_shape=(2, 2, 2, 1, 1, 1),
        expected_values=[
            ((0, 0, 0, 0, 0, 0), 0.4447744656845183),
            ((1, 1, 1, 0, 0, 0), 0.27769726675615586),
        ],
    ),
    _TestCase(
        g1=sf.GaussianBasis3d(
            max_degree=0,
            exponent=0.5,
            center=np.array([-2.0, 0.0, 1.0]),
        ),
        g2=sf.GaussianBasis3d(
            max_degree=1,
            exponent=0.2,
            center=np.array([1.0, 2.0, -1.0]),
        ),
        expected_shape=(1, 1, 1, 2, 2, 2),
        expected_values=[
            ((0, 0, 0, 0, 0, 0), 0.4447744656845183),
            ((0, 0, 0, 1, 1, 1), -4.339019793064935),
        ],
    ),
    _TestCase(
        g1=sf.GaussianBasis3d(
            max_degree=1,
            exponent=0.5,
            center=np.array([-2.0, 0.0, 1.0]),
        ),
        g2=sf.GaussianBasis3d(
            max_degree=1,
            exponent=0.2,
            center=np.array([1.0, 2.0, -1.0]),
        ),
        expected_shape=(2, 2, 2, 2, 2, 2),
        expected_values=[
            ((0, 0, 0, 0, 0, 0), 0.4447744656845183),
            ((1, 1, 1, 0, 0, 0), 0.27769726675615586),
            ((0, 0, 0, 1, 1, 1), -4.339019793064935),
            ((1, 1, 1, 1, 1, 1), 0.11032576884112527),
            ((1, 0, 1, 0, 1, 0), -0.69424316689038956),
            ((0, 1, 0, 1, 0, 1), 1.73560791722597396),
        ],
    ),
    _TestCase(
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
            ((0, 0, 0, 0, 0, 0), 0.4447744656845183),
            ((1, 1, 1, 0, 0, 0), 0.27769726675615586),
            ((2, 2, 2, 0, 0, 0), -0.44676243789090492),
        ],
    ),
    _TestCase(
        g1=sf.GaussianBasis3d(
            max_degree=2,
            exponent=0.5,
            center=np.array([-2.0, 0.0, 1.0]),
        ),
        g2=sf.GaussianBasis3d(
            max_degree=1,
            exponent=0.2,
            center=np.array([1.0, 2.0, -1.0]),
        ),
        expected_shape=(3, 3, 3, 2, 2, 2),
        expected_values=[
            ((0, 0, 0, 0, 0, 0), 0.4447744656845183),
            ((2, 1, 0, 1, 0, 1), 2.0360720605318074),
            ((0, 1, 2, 0, 1, 1), 0.28305920817531277),
            ((0, 0, 0, 1, 1, 1), -4.339019793064935),
            ((1, 1, 1, 1, 1, 1), 0.11032576884112527),
            ((2, 2, 2, 1, 1, 1), -0.7693172854848539),
        ],
    ),
    _TestCase(
        g1=sf.GaussianBasis3d(
            max_degree=2,
            exponent=0.5,
            center=np.array([-2.0, 0.0, 1.0]),
        ),
        g2=sf.GaussianBasis3d(
            max_degree=2,
            exponent=0.2,
            center=np.array([1.0, 2.0, -1.0]),
        ),
        expected_shape=(3, 3, 3, 3, 3, 3),
        expected_values=[
            ((0, 0, 0, 0, 0, 0), 0.4447744656845183),
            ((2, 1, 0, 1, 0, 1), 2.0360720605318074),
            ((0, 1, 2, 0, 1, 1), 0.28305920817531277),
            ((0, 0, 0, 1, 1, 1), -4.339019793064935),
            ((1, 1, 1, 1, 1, 1), 0.11032576884112527),
            ((2, 2, 2, 1, 1, 1), -0.7693172854848539),
            ((0, 0, 0, 2, 2, 2), -41.13682590933385796),
            ((1, 1, 1, 2, 2, 2), 1.14714848101974298),
            ((2, 2, 2, 2, 2, 2), 7.92099471611946715),
            ((0, 1, 2, 2, 1, 0), 2.65032668218701906),
        ],
    ),
    _TestCase(
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
            ((0, 0, 0, 0, 0, 0), 0.4447744656845183),
            ((2, 1, 0, 1, 0, 1), 2.0360720605318074),
            ((0, 1, 2, 0, 1, 1), 0.28305920817531277),
            ((0, 0, 0, 1, 1, 1), -4.339019793064935),
            ((1, 1, 1, 1, 1, 1), 0.11032576884112527),
            ((2, 2, 2, 1, 1, 1), -0.7693172854848539),
            ((0, 0, 0, 2, 2, 2), -41.13682590933385796),
            ((1, 1, 1, 2, 2, 2), 1.14714848101974298),
            ((2, 2, 2, 2, 2, 2), 7.92099471611946715),
            ((0, 1, 2, 2, 1, 0), 2.65032668218701906),
            ((0, 0, 0, 3, 3, 3), -501.21026299968980311),
            ((1, 1, 1, 3, 3, 3), 22.72527940186406781),
            ((2, 2, 2, 3, 3, 3), 13.10644008629708424),
            ((2, 1, 0, 1, 2, 3), -10.11178187978127418),
        ],
    ),
]


@pytest.mark.parametrize("case", _TEST_CASES)
def test_kinetic_3d(case):
    T = jit(sf.integrals.kinetic_3d)(case.g1, case.g2)
    assert T.shape == case.expected_shape

    for coords, expected in case.expected_values:
        actual = T[coords]
        print(coords)
        print(actual)
        print(expected)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
