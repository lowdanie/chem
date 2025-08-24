import dataclasses
import unittest

import numpy as np

from integrals import gaussian
from integrals import overlap


@dataclasses.dataclass
class _TestCase:
    g1: gaussian.CartesianGaussian1d
    g2: gaussian.CartesianGaussian1d
    degree1: int
    degree2: int
    expected: float


_TEST_CASES = [
    _TestCase(
        g1=gaussian.CartesianGaussian1d(max_degree=2, exponent=0.5, center=0.0),
        g2=gaussian.CartesianGaussian1d(max_degree=3, exponent=0.2, center=1.0),
        degree1=1,
        degree2=2,
        expected=-1.23145,
    ),
    # Integrate[(x - 0)^0 (x - 1)^0 E^(-0.5(x-0)^2) E^(-0.2(x-1)^2), {x, -Infinity, Infinity}]
    _TestCase(
        g1=gaussian.CartesianGaussian1d(max_degree=2, exponent=0.5, center=0.0),
        g2=gaussian.CartesianGaussian1d(max_degree=3, exponent=0.2, center=1.0),
        degree1=0,
        degree2=0,
        expected=1.83647,
    ),
]


class TestOverlap1D(unittest.TestCase):
    def test_overlap_1d_shape(self):
        g1 = gaussian.CartesianGaussian1d(
            exponent=0.5, center=0.0, max_degree=2
        )
        g2 = gaussian.CartesianGaussian1d(
            exponent=0.2, center=1.0, max_degree=3
        )
        S = overlap.overlap_1d(g1, g2)
        print(S)

        expected_shape = (g1.max_degree + 1, g2.max_degree + 1)
        self.assertEqual(S.shape, expected_shape)

    def test_overlap_1d_values(self):
        for case in _TEST_CASES:
            S = overlap.overlap_1d(case.g1, case.g2)
            actual = S[case.degree1, case.degree2]
            self.assertAlmostEqual(actual, case.expected, places=4)


if __name__ == "__main__":
    unittest.main()
