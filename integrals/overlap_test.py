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
    # The expected values were computed using Wolfram Alpha:
    # Integrate[(x+2)^i (x-1)^j E^(-0.5(x+2)^2) E^(-0.2(x-1)^2), {x, -Infinity, Infinity}]
    _TestCase1d(
        g1=gaussian.GaussianBasis1d(max_degree=2, exponent=0.5, center=-2.0),
        g2=gaussian.GaussianBasis1d(max_degree=3, exponent=0.2, center=1.0),
        expected=np.array(
            [
                [0.585662, -1.25499, 3.1076, -8.45198],
                [0.501996, -0.657376, 0.87081, -0.585418],
                [0.848613, -1.10132, 2.02701, -4.67289],
            ]
        ),
    ),
    _TestCase1d(
        g1=gaussian.GaussianBasis1d(max_degree=0, exponent=0.5, center=-2.0),
        g2=gaussian.GaussianBasis1d(max_degree=0, exponent=0.2, center=1.0),
        expected=np.array([[0.585662]]),
    ),
    _TestCase1d(
        g1=gaussian.GaussianBasis1d(max_degree=0, exponent=0.5, center=-2.0),
        g2=gaussian.GaussianBasis1d(max_degree=2, exponent=0.2, center=1.0),
        expected=np.array([[0.585662, -1.25499, 3.1076]]),
    ),
    _TestCase1d(
        g1=gaussian.GaussianBasis1d(max_degree=2, exponent=0.5, center=-2.0),
        g2=gaussian.GaussianBasis1d(max_degree=0, exponent=0.2, center=1.0),
        expected=np.array([[0.585662], [0.501996], [0.848613]]),
    ),
]

_TEST_CASES_3D = [
    # The expected values were computed using Wolfram Alpha:
    # Integrate[(x+2)^i (x-1)^j E^(-0.5(x+2)^2) E^(-0.2(x-1)^2), {x, -Infinity, Infinity}]
    _TestCase3d(
        g1=gaussian.GaussianBasis3d(
            max_degree=2,
            exponent=0.5,
            center=np.array([-2.0, 0.0, 1.0]),
        ),
        g2=gaussian.GaussianBasis3d(
            max_degree=3,
            exponent=0.2,
            center=np.array([1.0, 0.0, -1.0]),
        ),
        coords=np.array([2, 1, 0, 1, 2, 0]),
        expected=6.28319,
    ),
    _TestCase3d(
        g1=gaussian.GaussianBasis3d(
            max_degree=2,
            exponent=0.5,
            center=np.array([-2.0, 0.0, 1.0]),
        ),
        g2=gaussian.GaussianBasis3d(
            max_degree=3,
            exponent=0.2,
            center=np.array([1.0, 0.0, -1.0]),
        ),
        coords=np.array([0, 0, 0, 0, 0, 0]),
        expected=0.585662 * 3.54491,
    ),
]


class TestOverlap1D(unittest.TestCase):
    def test_overlap_1d(self):
        for case in _TEST_CASES_1D:
            S = overlap.overlap_1d(case.g1, case.g2)

            self.assertTrue(
                np.allclose(S, case.expected, atol=1e-4),
                msg=f"Failed for g1={case.g1}, g2={case.g2}. Got {S}, expected {case.expected}",
            )


if __name__ == "__main__":
    unittest.main()
