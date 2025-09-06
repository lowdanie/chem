import dataclasses
import unittest

import numpy as np

from integrals import coulomb
from integrals import gaussian


@dataclasses.dataclass
class _BoysTestCase:
    n: int
    x: float
    expected: float


@dataclasses.dataclass
class OneElectronTestCase:
    g1: coulomb.gaussian.GaussianBasis3d
    g2: coulomb.gaussian.GaussianBasis3d
    C: np.ndarray
    expected_shape: tuple[int, int, int, int, int, int]
    expected_values: dict[tuple[int, int, int, int, int, int], float]


@dataclasses.dataclass
class TwoElectronTestCase:
    g1: coulomb.gaussian.GaussianBasis3d
    g2: coulomb.gaussian.GaussianBasis3d
    g3: coulomb.gaussian.GaussianBasis3d
    g4: coulomb.gaussian.GaussianBasis3d
    expected_shape: tuple[int, ...]
    expected_values: dict[tuple[int, ...], float]


# The expected values were computed using sympy:
# >>> import sympy as sp
# >>> t = sp.symbols('t')
# >>> integrand = t**(2*n) * sp.exp(-x*t**2)
# >>> res = sp.integrate(integrand, (t, 0, 1))
# >>> expected = float(res.evalf())
_BOYS_TEST_CASES = [
    _BoysTestCase(n=0, x=0.0, expected=1.0),
    _BoysTestCase(n=0, x=1.0, expected=0.746824132812427),
    _BoysTestCase(n=1, x=0.0, expected=0.3333333333333333),
    _BoysTestCase(n=1, x=1.0, expected=0.18947234582049235),
    _BoysTestCase(n=1, x=10.0, expected=0.014010099528844014),
    _BoysTestCase(n=5, x=0.0, expected=0.09090909090909091),
    _BoysTestCase(n=5, x=1.0, expected=0.03936486451348416),
]

# The expected values were computed with scipy.integrate:
# >>> import numpy as np
# >>> from scipy import integrate
# >>> a, A = g1.exponent, g1.center
# >>> b, B = g2.exponent, g2.center
# >>>
# >>> def f(z,y,x):
# >>>   G1 = (x - A[0])**coords[0] * (y - A[1])**coords[1] * (z - A[2])**coords[2] *\
# >>>       np.exp(-a * ((x - A[0])**2 + (y - A[1])**2 + (z - A[2])**2))
# >>>   G2 = (x - B[0])**coords[3] * (y - B[1])**coords[4] * (z - B[2])**coords[5] *\
# >>>   np.exp(-b * ((x - B[0])**2 + (y - B[1])**2 + (z - B[2])**2))
# >>>   dist = np.sqrt((x - C[0])**2 + (y - C[1])**2 + (z - C[2])**2)
# >>>   return G1 * G2 / dist
# >>> expected, _ = integrate.nquad(f, -6, 6, -6, 6, -6, 6)
_ONE_ELECTRON_TEST_CASES = [
    OneElectronTestCase(
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
        C=np.array([1.0, -1.0, 2.0]),
        expected_shape=(3, 3, 3, 4, 4, 4),
        expected_values={
            (0, 0, 0, 0, 0, 0): 0.2714497857051819,
            (2, 2, 2, 0, 0, 0): 0.3617406824211391,
            (0, 0, 0, 3, 3, 3): 142.39866470590218,
            (2, 2, 2, 3, 3, 3): 37.41799405120356,
            (0, 1, 2, 1, 2, 3): 1.6977878772651622,
            (1, 0, 2, 2, 1, 1): -0.6010626174920991,
            (1, 1, 1, 1, 1, 1): -0.013057308270078337,
            (0, 0, 0, 0, 0, 3): 1.8621493778032885,
        },
    ),
]

_TWO_ELECTRON_TEST_CASES = [
    TwoElectronTestCase(
        g1=gaussian.GaussianBasis3d(
            max_degree=2,
            exponent=0.1,
            center=np.array([-2.0, 0.0, 1.0]),
        ),
        g2=gaussian.GaussianBasis3d(
            max_degree=2,
            exponent=0.2,
            center=np.array([1.0, 2.0, -1.0]),
        ),
        g3=gaussian.GaussianBasis3d(
            max_degree=2,
            exponent=0.3,
            center=np.array([2.0, 1.0, -1.0]),
        ),
        g4=gaussian.GaussianBasis3d(
            max_degree=2,
            exponent=0.4,
            center=np.array([0.0, -1.0, 2.0]),
        ),
        expected_shape=(3,) * 12,
        expected_values={
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0): 2.2635736015604326,
            (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0): 4.887010985110651,
            (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0): -1.9037098195706461,
            (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0): -2.741168594064688,
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0): 1.7859786090561764,
            (1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0): -0.9776795714262435,
            (1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0): -5.643744873448038,
            (1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0): 9.677117184425429,
            (1, 0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0): 0.4051466103984736,
            (0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0): -2.5782646240930593,
            (0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0): 0.9419537701948736,
            (0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1): -1.2763270734987182,
            (1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1): -0.046193719169573925,
            (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2): 1576.673145693577,
        },
    ),
]


class CoulombTest(unittest.TestCase):
    def test_boys(self):
        for case in _BOYS_TEST_CASES:
            actual = coulomb.boys(case.n, case.x)
            self.assertAlmostEqual(
                actual,
                case.expected,
                places=12,
            )

    def test_one_electron(self):
        for case in _ONE_ELECTRON_TEST_CASES:
            I = coulomb.one_electron(case.g1, case.g2, case.C)
            self.assertEqual(I.shape, case.expected_shape)
            for coord, expected in case.expected_values.items():
                actual = I[tuple(coord)]
                self.assertTrue(
                    np.allclose(actual, expected), msg=f"coords={coord}"
                )

    def test_two_electron(self):
        for case in _TWO_ELECTRON_TEST_CASES:
            I = coulomb.two_electron(case.g1, case.g2, case.g3, case.g4)
            self.assertEqual(I.shape, case.expected_shape)
            for coord, expected in case.expected_values.items():
                actual = I[tuple(coord)]
                self.assertTrue(
                    np.allclose(actual, expected),
                    msg=f"actual={actual}, expected={expected}, coords={coord}",
                )


if __name__ == "__main__":
    unittest.main()
