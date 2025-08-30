import dataclasses
import unittest

import numpy as np

from integrals import coulomb


@dataclasses.dataclass
class _BoysTestCase:
    n: int
    x: float
    expected: float


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


class CoulombTest(unittest.TestCase):
    def test_boys(self):
        for case in _BOYS_TEST_CASES:
            actual = coulomb.boys(case.n, case.x)
            self.assertAlmostEqual(
                actual,
                case.expected,
                places=12,
            )


if __name__ == "__main__":
    unittest.main()
