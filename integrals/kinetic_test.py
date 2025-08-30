import dataclasses
import unittest

import numpy as np

from integrals import gaussian
from integrals import overlap
from integrals import kinetic


@dataclasses.dataclass
class _TestCase1d:
    g1: gaussian.GaussianBasis1d
    g2: gaussian.GaussianBasis1d
    expected: np.ndarray


@dataclasses.dataclass
class _TestCase3dSparse:
    g1: gaussian.GaussianBasis3d
    g2: gaussian.GaussianBasis3d
    expected_shape: tuple[int, int, int, int, int, int]
    coords: np.ndarray  # shape (n, 6)
    expected: np.ndarray  # shape (n,


_TEST_CASES_1D = [
    # The expected values were computed using sympy.
    # For a given i,j we compute:
    # >>> import sympy as sp
    # >>> x = sp.symbols('x')
    # >>> expected = np.zeros((d1+1, d2-1)) # where d1 = g1.max_degree, d2 = g2.max_degree
    # >>> for i in range(d1+1):
    # >>>     for j in range(d2-1):
    # >>>         integrand = (x+2)**i * sp.exp(-0.5*(x+2)**2) *\
    # >>>                     sp.diff((x-1)**j * sp.exp(-0.2*(x-1)**2), x, 2)
    # >>>         res = sp.integrate(integrand, (x, -sp.oo, sp.oo))
    # >>>         expected[i, j] = float(res.evalf())
    _TestCase1d(
        g1=gaussian.GaussianBasis1d(max_degree=2, exponent=0.5, center=-2.0),
        g2=gaussian.GaussianBasis1d(max_degree=3, exponent=0.2, center=1.0),
        expected=np.array(
            [
                [0.2629504373846241, 0.1536723335364682],
                [-0.06146893341458779, 0.6951843659983115],
                [-0.01512330901469758, 0.5739191232076303],
            ]
        ),
    ),
    _TestCase1d(
        g1=gaussian.GaussianBasis1d(max_degree=2, exponent=0.5, center=-2.0),
        g2=gaussian.GaussianBasis1d(max_degree=2, exponent=0.2, center=1.0),
        expected=np.array(
            [
                [0.2629504373846241],
                [-0.06146893341458779],
                [-0.01512330901469758],
            ]
        ),
    ),
    _TestCase1d(
        g1=gaussian.GaussianBasis1d(max_degree=2, exponent=0.5, center=-2.0),
        g2=gaussian.GaussianBasis1d(max_degree=1, exponent=0.2, center=1.0),
        expected=np.empty((3, 0)),
    ),
    _TestCase1d(
        g1=gaussian.GaussianBasis1d(max_degree=0, exponent=0.5, center=-2.0),
        g2=gaussian.GaussianBasis1d(max_degree=0, exponent=0.2, center=1.0),
        expected=np.empty((1, 0)),
    ),
]

_TEST_CASES_3D = [
    # The expected values were computed using sympy.
    # For a given array of coord of shape (6,), we compute the integral:
    # >>> import sympy as sp
    # >>> G1 = (x + 2)**coord[0] * (y - 0)**coord[1] * (z - 1)**coord[2] *\
    # >>>      sp.exp(-0.5 * ((x + 2)**2 + (y - 0)**2 + (z - 1)**2))
    # >>> G2 = (x - 1)**coord[3] * (y - 2)**coord[4] * (z + 1)**coord[5] *\
    # >>>      sp.exp(-0.2 * ((x - 1)**2 + (y - 2)**2 + (z + 1)**2))
    # >>> integrand = G1 * (sp.diff(G2, x, 2) +\
    # >>>                   sp.diff(G2, y, 2) +\
    # >>>                   sp.diff(G2, z, 2))
    # >>> res = sp.integrate(integrand,
    # >>>                    (x, -sp.oo, sp.oo),
    # >>>                    (y, -sp.oo, sp.oo),
    # >>>                    (z, -sp.oo, sp.oo))
    # >>> expected = float(res.evalf())
    _TestCase3dSparse(
        g1=gaussian.GaussianBasis3d(
            max_degree=2,
            exponent=0.5,
            center=np.array([-2.0, 0.0, 1.0]),  # -2,0,1
        ),
        g2=gaussian.GaussianBasis3d(
            max_degree=3,
            exponent=0.2,
            center=np.array([1.0, 2.0, -1.0]),  # 1,2,-1
        ),
        expected_shape=(3, 3, 3, 2, 2, 2),
        coords=np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [2, 1, 0, 1, 0, 1],
                [2, 2, 2, 1, 1, 1],
            ]
        ),
        expected=np.array(
            [
                0.44477446568451884,
                2.0360720605318163,
                -0.7693172854848808,
            ]
        ),
    )
]


class KineticTest(unittest.TestCase):
    def test_kinetic_1d_from_overlap_1d(self):
        for case in _TEST_CASES_1D:
            S = overlap.overlap_1d(case.g1, case.g2)
            T = kinetic.kinetic_1d_from_overlap_1d(S, case.g1, case.g2)

            self.assertTrue(
                np.allclose(T, case.expected, atol=1e-7),
                msg=f"Failed for g1={case.g1}, g2={case.g2}. Got {T}, expected {case.expected}",
            )

    def test_kinetic_3d_from_1d(self):
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

            T = kinetic.kinetic_3d_from_overlap_1d(
                S_x, S_y, S_z, case.g1, case.g2
            )
            self.assertEqual(T.shape, case.expected_shape)
            for coord, expected in zip(case.coords, case.expected):
                actual = T[tuple(coord)]
                self.assertTrue(
                    np.allclose(actual, expected, atol=1e-7),
                    msg=f"Failed for g1={case.g1}, g2={case.g2}, coord={coord}. Got {actual}, expected {expected}",
                )


if __name__ == "__main__":
    unittest.main()
