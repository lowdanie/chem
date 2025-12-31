import dataclasses
import pytest

import numpy as np
import numpy.typing as npt

import slaterform as sf


@dataclasses.dataclass
class _EvaluateDensityTestCase:
    basis: sf.BatchedBasis
    P: np.ndarray  # shape: (n_basis, n_basis)
    grid: sf.RegularGrid
    expected_density: np.ndarray  # shape: grid.dims


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
            basis=sf.BatchedBasis(
                atoms=[],
                basis_blocks=[
                    sf.BasisBlock(
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
                block_starts=np.array([0]),
                batches_1e=[],
                batches_2e=[],
            ),
            P=np.array([[1.0, 2.0], [3.0, 4.0]]),
            grid=sf.RegularGrid(
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
    density_values = sf.analysis.evaluate_density(case.basis, case.P, case.grid)
    np.testing.assert_allclose(density_values, case.expected_density)
