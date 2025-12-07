import dataclasses
import unittest

import numpy as np

from basis import cartesian


@dataclasses.dataclass
class _GenerateCartesianPowersTestCase:
    l: int
    expected: np.ndarray


_GENERATE_CARTESIAN_POWERS_TEST_CASES = [
    _GenerateCartesianPowersTestCase(
        l=0,
        expected=np.array([[0, 0, 0]]),
    ),
    _GenerateCartesianPowersTestCase(
        l=1,
        expected=np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        ),
    ),
]


class CartesianTest(unittest.TestCase):
    def test_generate_cartesian_powers(self):
        for case in _GENERATE_CARTESIAN_POWERS_TEST_CASES:
            powers = cartesian.generate_cartesian_powers(case.l)

            self.assertTrue(
                np.array_equal(powers, case.expected),
                msg=f"Failed for l={case.l}. Got {powers}, expected {case.expected}",
            )

    def test_generate_cartesian_powers_invalid_l(self):
        with self.assertRaises(ValueError):
            cartesian.generate_cartesian_powers(2)
