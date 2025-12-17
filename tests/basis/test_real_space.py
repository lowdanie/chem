import dataclasses
import pytest

import numpy as np

from hflib.basis import basis_block
from hflib.basis import real_space


@dataclasses.dataclass
class _EvaluateTestCase:
    basis_block: basis_block.BasisBlock
    points: np.ndarray  # shape (..., 3)
    expected: np.ndarray  # shape (..., n_basis)


# The test cases were generated using the following code:
# import numpy as np
#
# def evaluate_gto(r, center, powers, exponents, coeffs):
#     diff = r - center
#     dist_sq = np.dot(diff, diff)
#
#     angular = np.prod(np.power(diff, powers))
#     radial = np.dot(coeffs, np.exp(-exponents * dist_sq))
#
#     return angular * radial
#
# batch_shape = points.shape[:-1]
# n_cart = contraction_matrix.shape[0]
# n_basis = basis_transform.shape[0]
#
# expected_shape = batch_shape + (n_basis,)
# block_eval = np.zeros(expected_shape, dtype=np.float64)
#
# for idx in np.ndindex(batch_shape):
#     r = points[idx]
#     eval_cart = np.zeros((n_cart,), dtype=np.float64)
#
#     for i in range(n_cart):
#         powers = cartesian_powers[i]
#         coeffs = contraction_matrix[i]
#         eval_cart[i] = evaluate_gto(r, center, powers, exponents, coeffs)
#
#     block_eval[idx] = np.matmul(basis_transform, eval_cart)
@pytest.mark.parametrize(
    "case",
    [
        _EvaluateTestCase(
            basis_block=basis_block.BasisBlock(
                center=np.array([1.0, -1.0, 0.0]),
                exponents=np.array([0.1, 0.2, 0.3]),
                cartesian_powers=np.array(
                    [
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                    ]
                ),
                contraction_matrix=np.array(
                    [
                        [0.1, 0.2, 0.3],
                        [0.2, 0.3, 0.4],
                        [0.3, 0.4, 0.5],
                        [0.4, 0.5, 0.6],
                    ]
                ),
                basis_transform=np.diag([1.0, 2.0, 3.0, 4.0]),
            ),
            points=np.array(
                [
                    [
                        [-3.0, -2.64705882, -2.29411765],
                        [-1.94117647, -1.58823529, -1.23529412],
                        [-0.88235294, -0.52941176, -0.17647059],
                    ],
                    [
                        [0.17647059, 0.52941176, 0.88235294],
                        [1.23529412, 1.58823529, 1.94117647],
                        [2.29411765, 2.64705882, 3.0],
                    ],
                ]
            ),
            expected=np.array(
                [
                    [
                        [0.01097335, -0.16775453, -0.15300494, -0.37587975],
                        [0.07206571, -0.72604674, -0.30845326, -1.11745901],
                        [0.25809083, -1.52595817, 0.7801051, -0.49398794],
                    ],
                    [
                        [0.25809083, -0.6676067, 2.53534156, 2.4699397],
                        [0.07206571, 0.05808374, 1.35719435, 1.75600701],
                        [0.01097335, 0.05427352, 0.33879665, 0.49153505],
                    ],
                ]
            ),
        ),
        _EvaluateTestCase(
            basis_block=basis_block.BasisBlock(
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
                basis_transform=np.array([[1.0], [2.0]]),
            ),
            points=np.array([[-3.0, -1.8, -0.6], [0.6, 1.8, 3.0]]),
            expected=np.array(
                [[0.07120274, 0.14240548], [0.07120274, 0.14240548]]
            ),
        ),
        _EvaluateTestCase(
            basis_block=basis_block.BasisBlock(
                center=np.array([1.0, -1.0, 0.0]),
                exponents=np.array([0.1, 0.2, 0.3]),
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
                basis_transform=np.diag([2.0, 3.0]),
            ),
            points=np.array(
                [
                    [
                        [
                            [-3.0, -2.73913043, -2.47826087],
                            [-2.2173913, -1.95652174, -1.69565217],
                        ],
                        [
                            [-1.43478261, -1.17391304, -0.91304348],
                            [-0.65217391, -0.39130435, -0.13043478],
                        ],
                    ],
                    [
                        [
                            [0.13043478, 0.39130435, 0.65217391],
                            [0.91304348, 1.17391304, 1.43478261],
                        ],
                        [
                            [1.69565217, 1.95652174, 2.2173913],
                            [2.47826087, 2.73913043, 3.0],
                        ],
                    ],
                ]
            ),
            expected=np.array(
                [
                    [
                        [[0.01906884, -0.32507164], [0.08089177, -1.00157908]],
                        [[0.28243461, -2.33814124], [0.59638968, -3.1243734]],
                    ],
                    [
                        [[0.59638968, -1.64440705], [0.28243461, -0.08350504]],
                        [[0.08089177, 0.21655764], [0.01906884, 0.12013517]],
                    ],
                ]
            ),
        ),
    ],
)
def test_evaluate(case):
    result = real_space.evaluate(case.basis_block, case.points)
    np.testing.assert_allclose(
        result,
        case.expected,
        rtol=1e-7,
        atol=1e-7,
    )
