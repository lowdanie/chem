import dataclasses
import pytest

import numpy as np

from chem.integrals import gaussian
from chem.integrals import overlap


@dataclasses.dataclass
class _TestCase1d:
    g1: gaussian.GaussianBasis1d
    g2: gaussian.GaussianBasis1d
    expected: np.ndarray


@dataclasses.dataclass
class _TestCase3dSparse:
    g1: gaussian.GaussianBasis3d
    g2: gaussian.GaussianBasis3d
    expected_indices: np.ndarray  # shape (n, 6)
    expected_values: np.ndarray  # shape (n,)


# The expected values were computed using sympy.
# For a given i,j we compute:
# >>> import sympy as sp
# >>> x = sp.symbols('x')
# >>> integrand = (x+2)**i * (x-1)**j * \
# ...             sp.exp(-0.5*(x+2)**2) * sp.exp(-0.2*(x-1)**2)
# >>> res = sp.integrate(integrand, (x, -sp.oo, sp.oo))
# >>> expected[i, j] = float(res.evalf())
_TEST_CASES_1D = [
    _TestCase1d(
        g1=gaussian.GaussianBasis1d(max_degree=2, exponent=0.5, center=-2.0),
        g2=gaussian.GaussianBasis1d(max_degree=3, exponent=0.2, center=1.0),
        expected=np.array(
            [
                [0.58566234, -1.25499072, 3.10759608, -8.45197834],
                [0.50199629, -0.65737609, 0.87080989, -0.58541841],
                [0.84861278, -1.10131839, 2.02701126, -4.67289341],
            ]
        ),
    ),
    _TestCase1d(
        g1=gaussian.GaussianBasis1d(max_degree=0, exponent=0.5, center=-2.0),
        g2=gaussian.GaussianBasis1d(max_degree=0, exponent=0.2, center=1.0),
        expected=np.array([[0.58566234]]),
    ),
    _TestCase1d(
        g1=gaussian.GaussianBasis1d(max_degree=0, exponent=0.5, center=-2.0),
        g2=gaussian.GaussianBasis1d(max_degree=2, exponent=0.2, center=1.0),
        expected=np.array([[0.58566234, -1.25499072, 3.10759608]]),
    ),
    _TestCase1d(
        g1=gaussian.GaussianBasis1d(max_degree=2, exponent=0.5, center=-2.0),
        g2=gaussian.GaussianBasis1d(max_degree=0, exponent=0.2, center=1.0),
        expected=np.array([[0.58566234], [0.50199629], [0.84861278]]),
    ),
]

# The expected values were computed using sympy.
# def build_cartesian_gaussian(coords, center, exponent, powers):
#     x, y, z = coords
#     cx, cy, cz = center
#     ix, iy, iz = powers
#
#     monom = (x - cx) ** ix * (y - cy) ** iy * (z - cz) ** iz
#     dist_sq = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
#     return monom * sp.exp(-exponent * dist_sq)
#
# powers_array1 = expected_indices[:, :3]
# powers_array2 = expected_indices[:, 3:]
#
# coords = sp.symbols("x,y,z")
# x, y, z = coords
#
# expected_values = []
#
# for powers1, powers2 in zip(powers_array1, powers_array2):
#     cg1 = build_cartesian_gaussian(
#         coords,
#         g1.center,
#         g1.exponent,
#         powers1,
#     )
#     cg2 = build_cartesian_gaussian(
#         coords,
#         g2.center,
#         g2.exponent,
#         powers2,
#     )
#     integrand = cg1 * cg2
#
#     res = sp.integrate(
#         integrand,
#         (x, -sp.oo, sp.oo),
#         (y, -sp.oo, sp.oo),
#         (z, -sp.oo, sp.oo),
#     )
#     output.append(float(res.evalf()))s
_TEST_CASES_3D = [
    _TestCase3dSparse(
        g1=gaussian.GaussianBasis3d(
            max_degree=2,
            exponent=0.5,
            center=np.array([-2.0, 0.0, 1.0]),
        ),
        g2=gaussian.GaussianBasis3d(
            max_degree=3,
            exponent=0.2,
            center=np.array([1.0, 2.0, -1.0]),
        ),
        expected_indices=np.array(
            [
                [2, 1, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0],
                [2, 2, 2, 3, 3, 3],
            ]
        ),
        expected_values=np.array(
            [
                -1.28674380,
                0.83822880,
                98.95957347,
            ]
        ),
    )
]


@pytest.mark.parametrize("case", _TEST_CASES_1D)
def test_overlap_1d(case):
    S = overlap.overlap_1d(case.g1, case.g2)

    np.testing.assert_allclose(S, case.expected, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("case", _TEST_CASES_3D)
def test_overlap_3d_from_1d(case):
    S_x = overlap.overlap_1d(
        gaussian.gaussian_3d_to_1d(case.g1, 0),
        gaussian.gaussian_3d_to_1d(case.g2, 0),
    )
    S_y = overlap.overlap_1d(
        gaussian.gaussian_3d_to_1d(case.g1, 1),
        gaussian.gaussian_3d_to_1d(case.g2, 1),
    )
    S_z = overlap.overlap_1d(
        gaussian.gaussian_3d_to_1d(case.g1, 2),
        gaussian.gaussian_3d_to_1d(case.g2, 2),
    )

    S = overlap.overlap_3d_from_1d(S_x, S_y, S_z)

    indices = tuple(case.expected_indices.T)
    actual_values = S[indices]
    np.testing.assert_allclose(
        actual_values, case.expected_values, atol=1e-7, rtol=1e-7
    )


@pytest.mark.parametrize("case", _TEST_CASES_3D)
def test_overlap_3d(case):
    S = overlap.overlap_3d(case.g1, case.g2)

    indices = tuple(case.expected_indices.T)
    actual_values = S[indices]
    np.testing.assert_allclose(
        actual_values, case.expected_values, atol=1e-7, rtol=1e-7
    )
