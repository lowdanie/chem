import dataclasses
import pytest

import numpy as np
import numpy.typing as npt

from slaterform.analysis import density
from slaterform.analysis import grid as analysis_grid
from slaterform.basis import basis_block
from slaterform.structure import atom
from slaterform.structure import molecular_basis


@dataclasses.dataclass
class _EvaluateDensityTestCase:
    mol_basis: molecular_basis.MolecularBasis
    P: npt.NDArray[np.float64]  # shape: (n_basis, n_basis)
    grid: analysis_grid.RegularGrid
    expected_density: npt.NDArray[np.float64]  # shape: grid.dims


# The basis functions were evaluated using the same method as in
# tests/basis/test_real_space.py
#
# The density values were computed with the following script:
# for idx in np.ndindex(batch_shape):
#     rho = block_eval[idx]
#     density[idx] = np.dot(rho, np.matmul(P, rho))
@pytest.mark.parametrize(
    "case",
    [
        _EvaluateDensityTestCase(
            mol_basis=molecular_basis.MolecularBasis(
                atoms=[],
                basis_blocks=[
                    basis_block.BasisBlock(
                        center=np.array([0.0, -1.0, 1.0]),
                        exponents=np.array([0.4, 0.5, 0.6]),
                        cartesian_powers=np.array([[0, 0, 0], [1, 0, 0]]),
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
            P=np.array([[1.0, 2.0], [3.0, 4.0]]),
            grid=analysis_grid.RegularGrid(
                origin=np.array([-1.0, -1.0, 1.0]),
                spacing=1.0,
                dims=(2, 2, 2),
            ),
            expected_density=np.array(
                [
                    [[0.919942, 0.34029037], [0.34029037, 0.12753947]],
                    [[0.36, 0.12459603], [0.12459603, 0.04362544]],
                ]
            ),
        ),
    ],
)
def test_evaluate_density(
    case: _EvaluateDensityTestCase,
) -> None:
    density_values = density.evaluate(case.mol_basis, case.P, case.grid)
    np.testing.assert_allclose(density_values, case.expected_density)
