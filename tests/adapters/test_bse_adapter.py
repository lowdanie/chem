import dataclasses
import pytest

import numpy as np

from slaterform.adapters import bse
from slaterform.basis import contracted_gto


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
    _TEST_CASE(
        basis_name="sto-3g",
        element=8,
        expected_gtos=[
            contracted_gto.ContractedGTO(
                primitive_type=contracted_gto.PrimitiveType.CARTESIAN,
                angular_momentum=(0,),
                exponents=np.array(
                    [0.1307093214e03, 0.2380886605e02, 0.6443608313e01],
                    dtype=np.float64,
                ),
                coefficients=np.array(
                    [[0.1543289673, 0.5353281423, 0.4446345422]],
                    dtype=np.float64,
                ),
            ),
            contracted_gto.ContractedGTO(
                primitive_type=contracted_gto.PrimitiveType.CARTESIAN,
                angular_momentum=(0, 1),
                exponents=np.array(
                    [0.5033151319e01, 0.1169596125e01, 0.3803889600],
                    dtype=np.float64,
                ),
                coefficients=np.array(
                    [
                        [-0.9996722919e-01, 0.3995128261, 0.7001154689],
                        [0.1559162750, 0.6076837186, 0.3919573931],
                    ],
                    dtype=np.float64,
                ),
            ),
        ],
    ),
]


@pytest.mark.parametrize("case", _TEST_CASES)
def test_load(case: _TEST_CASE):
    actual_gtos = bse.load(case.basis_name, case.element)

    assert len(actual_gtos) == len(case.expected_gtos), (
        f"Number of GTOS mismatch for basis {case.basis_name}, element {case.element}. "
        f"Got {len(actual_gtos)}, expected {len(case.expected_gtos)}"
    )

    for actual, expected in zip(actual_gtos, case.expected_gtos):
        assert actual.primitive_type == expected.primitive_type
        assert actual.angular_momentum == expected.angular_momentum
        np.testing.assert_allclose(actual.exponents, expected.exponents)
        np.testing.assert_allclose(actual.coefficients, expected.coefficients)
