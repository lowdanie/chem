import dataclasses
import unittest

import numpy as np

from basis import bse_adapter
from basis import contracted_gto


@dataclasses.dataclass
class _TEST_CASE:
    basis_name: str
    element: int
    expected_gtos: list[contracted_gto.ContractedGTO]


_TEST_CASES = [
    _TEST_CASE(
        basis_name="sto-3g",
        element=1,
        expected_gtos=[
            contracted_gto.ContractedGTO(
                primitive_type=contracted_gto.PrimitiveType.CARTESIAN,
                angular_momentum=(0,),
                exponents=np.array(
                    [3.425250914, 0.6239137298, 0.1688554040], dtype=np.float64
                ),
                coefficients=np.array(
                    [[0.1543289673, 0.5353281423, 0.4446345422]],
                    dtype=np.float64,
                ),
            )
        ],
    ),
]


class BseAdaptorTest(unittest.TestCase):
    def assertGtoEqual(
        self,
        actual: contracted_gto.ContractedGTO,
        expected: contracted_gto.ContractedGTO,
    ):
        self.assertEqual(actual.primitive_type, expected.primitive_type)
        self.assertEqual(actual.angular_momentum, expected.angular_momentum)
        self.assertTrue(
            np.allclose(actual.exponents, expected.exponents),
            msg=f"Exponents mismatch. Got {actual.exponents}, expected {expected.exponents}",
        )
        self.assertTrue(
            np.allclose(actual.coefficients, expected.coefficients),
            msg=f"Coefficients mismatch. Got {actual.coefficients}, expected {expected.coefficients}",
        )

    def test_load(self):
        for case in _TEST_CASES:
            actual_gtos = bse_adapter.load(case.basis_name, case.element)

            self.assertEqual(
                len(actual_gtos),
                len(case.expected_gtos),
                msg=f"Number of GTOS mismatch for basis {case.basis_name}, element {case.element}. "
                f"Got {len(actual_gtos)}, expected {len(case.expected_gtos)}",
            )

            for actual, expected in zip(actual_gtos, case.expected_gtos):
                self.assertGtoEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
