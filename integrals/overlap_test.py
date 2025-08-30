import dataclasses
import unittest

import numpy as np

from integrals import gaussian
from integrals import overlap


@dataclasses.dataclass
class _TestCase1d:
    g1: gaussian.GaussianBasis1d
    g2: gaussian.GaussianBasis1d
    expected: np.ndarray


@dataclasses.dataclass
class _TestCase3dSparse:
    g1: gaussian.GaussianBasis3d
    g2: gaussian.GaussianBasis3d
    coords: np.ndarray  # shape (n, 6)
    expected: np.ndarray  # shape (n,)


_TEST_CASES_1D = [
    # The expected values were computed using sympy.
    # For a given i,j we compute:
    # >>> import sympy as sp
    # >>> x = sp.symbols('x')
    # >>> integrand = (x+2)**i * (x-1)**j * \
    # ...             sp.exp(-0.5*(x+2)**2) * sp.exp(-0.2*(x-1)**2)
    # >>> res = sp.integrate(integrand, (x, -sp.oo, sp.oo))
    # >>> expected[i, j] = float(res.evalf())
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

_TEST_CASES_3D = [
    # The expected values were computed using sympy.
    # For a given array of coord of shape (6,), we compute the integral:
    # >>> import sympy as sp
    # >>> x, y, z = sp.symbols('x y z')
    # >>> c = sp.symbols('c_0:6')
    # >>> integrand = (x + 2)**c[0] * (y - 0)**c[0] * (z - 1)**c[2] *\
    # ...             (x - 1)**c[3] * (y - 2)**c[4] * (z + 1)**c[5] *\
    # ...             sp.exp(-0.5 * ((x + 2)**2 + y**2 + (z - 1)**2)) *\
    # ...             sp.exp(-0.2 * ((x - 1)**2 + (y - 2)**2 + (z + 1)**2))
    # >>> integrand_sub = integrand.subs(dict(zip(cs, coord)))
    # >>> res = sp.integrate(integrand_sub, (x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo), (z, -sp.oo, sp.oo))
    # >>> expected = float(res.evalf())
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
        coords=np.array(
            [
                [2, 1, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0],
                [2, 2, 2, 3, 3, 3],
            ]
        ),
        expected=np.array(
            [
                -1.28674380070202,
                0.838228800713131,
            ]
        ),
    )
]


class TestOverlap(unittest.TestCase):
    def test_overlap_1d(self):
        for case in _TEST_CASES_1D:
            S = overlap.overlap_1d(case.g1, case.g2)

            self.assertTrue(
                np.allclose(S, case.expected, atol=1e-7),
                msg=f"Failed for g1={case.g1}, g2={case.g2}. Got {S}, expected {case.expected}",
            )

    def test_overlap_3d_from_1d(self):
        for case in _TEST_CASES_3D:
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

            for coord, expected in zip(case.coords, case.expected):
                actual = S[tuple(coord)]
                self.assertTrue(
                    np.allclose(actual, expected, atol=1e-7),
                    msg=f"Failed for g1={case.g1}, g2={case.g2}, coord={coord}. Got {actual}, expected {expected}",
                )


if __name__ == "__main__":
    unittest.main()
