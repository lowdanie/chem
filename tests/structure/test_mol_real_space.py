import dataclasses
import pytest

import numpy as np
import numpy.typing as npt

from slaterform.basis import basis_block
from slaterform.structure import molecular_basis
from slaterform.structure import real_space


@dataclasses.dataclass
class _EvaluateTestCase:
    mol_basis: molecular_basis.MolecularBasis
    points: npt.NDArray[np.float64]  # shape (..., 3)
    expected: npt.NDArray[np.float64]  # shape (..., n_basis)


# The test data was generated using the same method as in
# tests/basis/test_real_space.py
@pytest.mark.parametrize(
    "case",
    [
        _EvaluateTestCase(
            mol_basis=molecular_basis.MolecularBasis(
                atoms=[],
                basis_blocks=[
                    basis_block.BasisBlock(
                        center=np.array([1.0, -1.0, 0.0]),
                        exponents=np.array([0.1, 0.2, 0.3]),
                        cartesian_powers=np.array(
                            [
                                [0, 0, 0],
                            ]
                        ),
                        contraction_matrix=np.array(
                            [
                                [0.3, 0.4, 0.5],
                            ]
                        ),
                        basis_transform=np.eye(1),
                    ),
                    basis_block.BasisBlock(
                        center=np.array([0.0, -1.0, 1.0]),
                        exponents=np.array([0.4, 0.5, 0.6]),
                        cartesian_powers=np.array(
                            [
                                [0, 0, 0],
                                [1, 0, 0],
                            ]
                        ),
                        contraction_matrix=np.array(
                            [
                                [0.1, 0.2, 0.3],
                                [0.3, 0.4, 0.5],
                            ]
                        ),
                        basis_transform=np.eye(2),
                    ),
                ],
            ),
            points=np.array(
                [
                    [
                        [-3.0, -2.45454545, -1.90909091],
                        [-1.36363636, -0.81818182, -0.27272727],
                    ],
                    [
                        [0.27272727, 0.81818182, 1.36363636],
                        [1.90909091, 2.45454545, 3.0],
                    ],
                ]
            ),
            expected=np.array(
                [
                    [
                        [3.99302153e-02, 5.32904904e-05, -4.36489457e-04],
                        [3.88421501e-01, 9.55424446e-02, -2.77455970e-01],
                    ],
                    [
                        [3.88421501e-01, 9.55424446e-02, 5.54911940e-02],
                        [3.99302153e-02, 5.32904904e-05, 2.77766018e-04],
                    ],
                ]
            ),
        ),
    ],
)
def test_evaluate(case: _EvaluateTestCase):
    actual = real_space.evaluate(case.mol_basis, case.points)
    np.testing.assert_allclose(actual, case.expected, rtol=1e-7, atol=1e-7)
