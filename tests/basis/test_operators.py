import dataclasses
import pytest
import functools

from jax import jit
import numpy as np

from slaterform.basis import basis_block
from slaterform.basis import operators
import slaterform as sf


@dataclasses.dataclass
class _OneElectronMatrixTestCase:
    block1: basis_block.BasisBlock
    block2: basis_block.BasisBlock
    expected: np.ndarray


@dataclasses.dataclass
class _OneElectronCoulombMatrixTestCase:
    block1: basis_block.BasisBlock
    block2: basis_block.BasisBlock
    C: np.ndarray  # shape (3,)
    expected: np.ndarray


@dataclasses.dataclass
class _TwoElectronCoulombMatrixTestCase:
    block1: basis_block.BasisBlock
    block2: basis_block.BasisBlock
    block3: basis_block.BasisBlock
    block4: basis_block.BasisBlock
    expected_coords: np.ndarray  # shape (n, 3)
    expected_values: np.ndarray  # shape (n,)


# The expected values were computed using the following code:
# import numpy as np
# import sympy as sp
#
#
# def build_gto(coords, center, exponents, coeffs, powers):
#     x, y, z = coords
#     ix, iy, iz = powers
#     cx, cy, cz = center
#
#     monom = (x - cx) ** ix * (y - cy) ** iy * (z - cz) ** iz
#     dist_sq = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
#     return monom * sum(
#         coeffs[i] * sp.exp(-exponents[i] * dist_sq)
#         for i in range(len(coeffs))
#     )
#
# coords = sp.symbols("x,y,z")
# x, y, z = coords
#
# output = np.zeros(
#     (cartesian_powers1.shape[0], cartesian_powers2.shape[0]), dtype=np.float64
# )
#
# for row in range(cartesian_powers1.shape[0]):
#     for col in range(cartesian_powers2.shape[0]):
#         gto1 = build_gto(
#             coords,
#             center1,
#             exponents1,
#             contraction_matrix1[row, :],
#             cartesian_powers1[row],
#         )
#         gto2 = build_gto(
#             coords,
#             center2,
#             exponents2,
#             contraction_matrix2[col, :],
#             cartesian_powers2[col],
#         )
#
#         integrand = gto1 * gto2
#
#         res = sp.integrate(
#             integrand,
#             (x, -sp.oo, sp.oo),
#             (y, -sp.oo, sp.oo),
#             (z, -sp.oo, sp.oo),
#         )
#         output[row, col] = float(res.evalf())
#
# output *= 2 * 3  # basis transforms
# print(np.array2string(output, precision=8, separator=", ", floatmode="fixed"))
@pytest.mark.parametrize(
    "case",
    [
        _OneElectronMatrixTestCase(
            block1=basis_block.BasisBlock(
                center=np.array([1.0, 0.0, 0.0]),
                exponents=np.array([0.1, 0.2, 0.3], dtype=np.float64),
                cartesian_powers=np.array(
                    [
                        [0, 0, 0],
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                    ],
                ),
                contraction_matrix=np.array(
                    [
                        [0.1, 0.2, 0.3],
                        [0.2, 0.3, 0.4],
                        [0.3, 0.4, 0.5],
                        [0.4, 0.5, 0.6],
                        [0.5, 0.6, 0.7],
                    ],
                    dtype=np.float64,
                ),
                basis_transform=3 * np.eye(5, dtype=np.float64),
            ),
            block2=basis_block.BasisBlock(
                center=np.array([0.0, 1.0, -1.0]),
                exponents=np.array([0.4, 0.5], dtype=np.float64),
                cartesian_powers=np.array(
                    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                ),
                contraction_matrix=np.array(
                    [
                        [0.5, 0.4],
                        [0.4, 0.3],
                        [0.3, 0.2],
                        [0.2, 0.1],
                    ],
                    dtype=np.float64,
                ),
                basis_transform=2 * np.eye(4, dtype=np.float64),
            ),
            expected=np.array(
                [
                    [22.05680328, 5.44876680, -3.93533442, 2.42190204],
                    [34.67845110, 8.16843462, -5.90027202, 3.63210948],
                    [-33.38910378, 22.73986176, 5.30523000, -3.25401768],
                    [42.53635362, 9.24250716, 21.04540776, 4.08907374],
                    [-51.68360346, -11.12857206, 8.02635096, 15.82833756],
                ],
                dtype=np.float64,
            ),
        ),
    ],
)
def test_overlap_matrix(case):
    kernel = jit(
        functools.partial(
            operators.one_electron_matrix, operator=sf.integrals.overlap_3d
        )
    )
    result = kernel(case.block1, case.block2)
    np.testing.assert_allclose(result, case.expected, rtol=1e-7, atol=1e-8)


# The expected values were computed using the following code:
# import numpy as np
# import sympy as sp
#
#
# def build_gto(coords, center, exponents, coeffs, powers):
#     x, y, z = coords
#     ix, iy, iz = powers
#     cx, cy, cz = center
#
#     monom = (x - cx) ** ix * (y - cy) ** iy * (z - cz) ** iz
#     dist_sq = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
#     return monom * sum(
#         coeffs[i] * sp.exp(-exponents[i] * dist_sq)
#         for i in range(len(coeffs))
#     )
#
# coords = sp.symbols("x,y,z")
# x, y, z = coords
#
# output = np.zeros(
#     (cartesian_powers1.shape[0], cartesian_powers2.shape[0]), dtype=np.float64
# )
#
# for row in range(cartesian_powers1.shape[0]):
#     for col in range(cartesian_powers2.shape[0]):
#         gto1 = build_gto(
#             coords,
#             center1,
#             exponents1,
#             contraction_matrix1[row, :],
#             cartesian_powers1[row],
#         )
#         gto2 = build_gto(
#             coords,
#             center2,
#             exponents2,
#             contraction_matrix2[col, :],
#             cartesian_powers2[col],
#         )
#
#        nabla_sq_gto2 = sp.diff(gto2, x, 2) + sp.diff(gto2, y, 2) + sp.diff(gto2, z, 2)
#       integrand = gto1 * nabla_sq_gto2
#
#         res = sp.integrate(
#             integrand,
#             (x, -sp.oo, sp.oo),
#             (y, -sp.oo, sp.oo),
#             (z, -sp.oo, sp.oo),
#         )
#         output[row, col] = float(res.evalf())
#
# print(np.array2string(output, precision=8, separator=", ", floatmode="fixed"))
@pytest.mark.parametrize(
    "case",
    [
        _OneElectronMatrixTestCase(
            block1=basis_block.BasisBlock(
                center=np.array([1.0, 0.0, 0.0]),
                exponents=np.array([0.1, 0.2], dtype=np.float64),
                cartesian_powers=np.array(
                    [
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                    ],
                ),
                contraction_matrix=np.array(
                    [
                        [0.1, 0.2],
                        [0.2, 0.3],
                        [0.3, 0.4],
                        [0.4, 0.5],
                    ],
                    dtype=np.float64,
                ),
                basis_transform=np.eye(4, dtype=np.float64),
            ),
            block2=basis_block.BasisBlock(
                center=np.array([0.0, 1.0, -1.0]),
                exponents=np.array([0.4, 0.5], dtype=np.float64),
                cartesian_powers=np.array(
                    [
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                    ],
                ),
                contraction_matrix=np.array(
                    [
                        [0.5, 0.4],
                        [0.4, 0.3],
                        [0.3, 0.2],
                        [0.2, 0.1],
                    ],
                    dtype=np.float64,
                ),
                basis_transform=np.eye(4, dtype=np.float64),
            ),
            expected=np.array(
                [
                    [-1.17980844, -0.47221436, 0.34077151, -0.20932867],
                    [2.71122305, -1.63406735, -0.57890582, 0.35439258],
                    [-3.81218549, -1.11283681, -1.67280452, -0.49098921],
                    [4.91314794, 1.42225456, -1.02492020, -1.33388159],
                ],
                dtype=np.float64,
            ),
        ),
    ],
)
def test_kinetic_matrix(case):
    kernel = jit(
        functools.partial(
            operators.one_electron_matrix,
            operator=sf.integrals.kinetic_3d,
        )
    )
    result = kernel(case.block1, case.block2)
    np.testing.assert_allclose(result, case.expected, rtol=1e-7, atol=1e-8)


# Test values were computed with the following script:
#
# import numpy as np
# import sympy as sp
# from scipy import integrate
#
# def evaluate_gto(r, center, powers, exponents, coeffs):
#   diff = r - center
#   dist_sq = np.dot(diff, diff)
#
#   angular = np.prod(np.power(diff, powers))
#   radial = np.dot(coeffs, np.exp(-exponents * dist_sq))
#
#   return angular * radial
#
# def coulomb(z, y, x, center1, powers1, exponents1, coeffs1, center2, powers2, exponents2, coeffs2, C):
#   r = np.array([x, y, z])
#   gto1 = evaluate_gto(r, center1, powers1, exponents1, coeffs1)
#   gto2 = evaluate_gto(r, center2, powers2, exponents2, coeffs2)
#   dist = np.linalg.norm(r - C)
#
#   return gto1 * gto2 / dist
#
# for row in range()
#
# output = np.zeros(
#     (cartesian_powers1.shape[0], cartesian_powers2.shape[0]), dtype=np.float64
# )
#
# ranges = [[-6, 6], [-6, 6], [-6, 6]]
#
# for row in range(cartesian_powers1.shape[0]):
#     for col in range(cartesian_powers2.shape[0]):
#         extra_args = (
#             center1, cartesian_powers1[row], exponents1, contraction_matrix1[row],
#             center2, cartesian_powers2[col], exponents2, contraction_matrix2[col],
#             coulomb_center)
#
#         res, error = integrate.nquad(coulomb, ranges, args=extra_args, opts={'epsrel': 1e-6, 'epsabs': 1e-6})
#         output[row, col] = res
#
# print(np.array2string(output, precision=8, separator=", ", floatmode="fixed"))
@pytest.mark.parametrize(
    "case",
    [
        _OneElectronCoulombMatrixTestCase(
            block1=basis_block.BasisBlock(
                center=np.array([1.0, 0.0, 0.0]),
                exponents=np.array([0.1, 0.2], dtype=np.float64),
                cartesian_powers=np.array(
                    [
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                    ],
                ),
                contraction_matrix=np.array(
                    [
                        [0.1, 0.2],
                        [0.2, 0.3],
                        [0.3, 0.4],
                        [0.4, 0.5],
                    ],
                    dtype=np.float64,
                ),
                basis_transform=np.eye(4, dtype=np.float64),
            ),
            block2=basis_block.BasisBlock(
                center=np.array([0.0, 1.0, -1.0]),
                exponents=np.array([0.4, 0.5], dtype=np.float64),
                cartesian_powers=np.array(
                    [
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                    ],
                ),
                contraction_matrix=np.array(
                    [
                        [0.5, 0.4],
                        [0.4, 0.3],
                        [0.3, 0.2],
                        [0.2, 0.1],
                    ],
                    dtype=np.float64,
                ),
                basis_transform=np.eye(4, dtype=np.float64),
            ),
            C=np.array([0.0, 1.0, 0.0], dtype=np.float64),
            expected=np.array(
                [
                    [1.74628771, 0.25087061, -0.18098078, 0.28896585],
                    [-2.45294460, 1.08244415, 0.23190209, -0.41254053],
                    [3.48036177, 0.44995941, 1.11598434, 0.58204351],
                    [-2.88126031, -0.44356595, 0.31953551, 0.77315663],
                ],
                dtype=np.float64,
            ),
        ),
    ],
)
def test_one_electron_coulomb_matrix(case):
    kernel = jit(
        functools.partial(
            operators.one_electron_matrix,
            operator=functools.partial(sf.integrals.one_electron, C=case.C),
        )
    )
    result = kernel(case.block1, case.block2)

    np.testing.assert_allclose(result, case.expected, rtol=1e-5, atol=1e-5)


# The test values were generated using the pyquante2 library:
# import numpy as np
# from pyquante2 import basis
# from pyquante2.ints import two as two_ints
#
# def compute_matrix_element(case, coords):
#     coeffs1 = case.block1.contraction_matrix[coords[0]]
#     coeffs2 = case.block2.contraction_matrix[coords[1]]
#     coeffs3 = case.block3.contraction_matrix[coords[2]]
#     coeffs4 = case.block4.contraction_matrix[coords[3]]
#
#     powers1 = case.block1.cartesian_powers[coords[0]]
#     powers2 = case.block2.cartesian_powers[coords[1]]
#     powers3 = case.block3.cartesian_powers[coords[2]]
#     powers4 = case.block4.cartesian_powers[coords[3]]
#
#     output = 0
#
#     for a1,c1 in zip(case.block1.exponents, coeffs1):
#         for a2,c2 in zip(case.block2.exponents, coeffs2):
#             for a3,c3 in zip(case.block3.exponents, coeffs3):
#                 for a4,c4 in zip(case.block4.exponents, coeffs4):
#                     c_prod = c1 * c2 * c3 * c4
#                     integral = two_ints.coulomb_repulsion(
#                         case.block1.center, 1, powers1, a1,
#                         case.block2.center, 1, powers2, a2,
#                         case.block3.center, 1, powers3, a3,
#                         case.block4.center, 1, powers4, a4
#                     )
#                     output += c_prod * integral
#
#     return output
#
# expected_values = []
# for coords in expected_coords:
#     expected_values.append(compute_matrix_element(case, coords))
@pytest.mark.parametrize(
    "case",
    [
        _TwoElectronCoulombMatrixTestCase(
            block1=basis_block.BasisBlock(
                center=np.array([1.0, 0.0, 0.0]),
                exponents=np.array([0.1, 0.2], dtype=np.float64),
                cartesian_powers=np.array(
                    [
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                    ],
                ),
                contraction_matrix=np.array(
                    [
                        [0.1, 0.2],
                        [0.2, 0.3],
                        [0.3, 0.4],
                        [0.4, 0.5],
                    ],
                    dtype=np.float64,
                ),
                basis_transform=np.eye(4, dtype=np.float64),
            ),
            block2=basis_block.BasisBlock(
                center=np.array([0.0, 1.0, -1.0]),
                exponents=np.array([0.3, 0.4], dtype=np.float64),
                cartesian_powers=np.array(
                    [
                        [0, 0, 0],
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                    ],
                ),
                contraction_matrix=np.array(
                    [
                        [0.1, 0.8],
                        [0.2, 0.7],
                        [0.3, 0.6],
                        [0.4, 0.5],
                        [0.3, 0.2],
                    ],
                    dtype=np.float64,
                ),
                basis_transform=np.eye(5, dtype=np.float64),
            ),
            block3=basis_block.BasisBlock(
                center=np.array([1.0, 0.0, -1.0]),
                exponents=np.array([0.5, 0.6], dtype=np.float64),
                cartesian_powers=np.array(
                    [
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                    ],
                ),
                contraction_matrix=np.array(
                    [
                        [0.5, 0.4],
                        [0.2, 0.3],
                        [0.1, 0.2],
                        [0.4, 0.3],
                    ],
                    dtype=np.float64,
                ),
                basis_transform=np.eye(4, dtype=np.float64),
            ),
            block4=basis_block.BasisBlock(
                center=np.array([-1.0, 1.0, 0.0]),
                exponents=np.array([0.4, 0.5], dtype=np.float64),
                cartesian_powers=np.array(
                    [
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                    ],
                ),
                contraction_matrix=np.array(
                    [
                        [0.1, 0.2],
                        [0.4, 0.3],
                        [0.5, 0.6],
                        [0.8, 0.7],
                    ],
                    dtype=np.float64,
                ),
                basis_transform=np.eye(4, dtype=np.float64),
            ),
            expected_coords=np.array(
                [
                    [0, 0, 0, 0],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [1, 1, 1, 1],
                    [1, 2, 1, 2],
                    [0, 1, 2, 3],
                    [3, 2, 1, 0],
                    [3, 4, 3, 3],
                ]
            ),
            expected_values=np.array(
                [
                    0.59120314,
                    -0.77406601,
                    0.61057719,
                    -0.27909564,
                    1.67844830,
                    0.60539463,
                    0.67677035,
                    -0.27733146,
                    0.07706643,
                    0.95529527,
                ]
            ),
        ),
    ],
)
def test_two_electron_matrix(case):
    kernel = jit(
        functools.partial(
            operators.two_electron_matrix,
            operator=sf.integrals.two_electron,
        )
    )
    result = kernel(case.block1, case.block2, case.block3, case.block4)
    indices = tuple(case.expected_coords.T)
    values = result[indices]
    np.testing.assert_allclose(
        values, case.expected_values, rtol=1e-7, atol=1e-8
    )


# Contracted GTOs in BSE basis sets should have a self overlap matrix equal
# to the identity matrix
@pytest.mark.parametrize(
    "basis_name, element",
    [
        ("sto-3g", 1),
        ("sto-3g", 6),
        ("sto-3g", 8),
    ],
)
def test_bse_identity(basis_name, element):
    gtos = bse.load(basis_name, element)
    center = np.array([0.0, 1.0, 2.0])

    for gto in gtos:
        block = basis_block.build_basis_block(gto, center)
        num_basis = block.basis_transform.shape[0]
        S = operators.one_electron_matrix(block, block, overlap.overlap_3d)

        np.testing.assert_allclose(S, np.eye(num_basis), rtol=1e-7, atol=1e-7)


def test_overlap_symmetry():
    block = basis_block.BasisBlock(
        center=np.array([0.0, 0.0, 0.0]),
        exponents=np.array([1.0, 2.0]),
        cartesian_powers=np.array([[0, 0, 0], [1, 0, 0]]),
        contraction_matrix=np.array([[0.6, 0.4], [0.3, 0.7]]),
        basis_transform=np.eye(2),
    )

    kernel = jit(
        functools.partial(
            operators.one_electron_matrix, operator=sf.integrals.overlap_3d
        )
    )
    S = kernel(block, block)
    np.testing.assert_allclose(S, S.T, rtol=1e-7, atol=1e-8)


def test_kinetic_symmetry():
    block = basis_block.BasisBlock(
        center=np.array([0.0, 0.0, 0.0]),
        exponents=np.array([1.0, 2.0]),
        cartesian_powers=np.array([[0, 0, 0], [1, 0, 0]]),
        contraction_matrix=np.array([[0.6, 0.4], [0.3, 0.7]]),
        basis_transform=np.eye(2),
    )

    kernel = jit(
        functools.partial(
            operators.one_electron_matrix, operator=sf.integrals.kinetic_3d
        )
    )
    T = kernel(block, block)
    np.testing.assert_allclose(T, T.T, rtol=1e-7, atol=1e-8)


def test_one_electron_symmetry():
    block = basis_block.BasisBlock(
        center=np.array([0.0, 0.0, 0.0]),
        exponents=np.array([1.0, 2.0]),
        cartesian_powers=np.array([[0, 0, 0], [1, 0, 0]]),
        contraction_matrix=np.array([[0.6, 0.4], [0.3, 0.7]]),
        basis_transform=np.eye(2),
    )
    C = np.array([1.0, 0.0, 0.0])

    kernel = jit(
        functools.partial(
            operators.one_electron_matrix,
            functools.partial(sf.integrals.one_electron, C=C),
        )
    )
    V = kernel(block, block)
    np.testing.assert_allclose(V, V.T, rtol=1e-7, atol=1e-8)


def test_two_electron_coulomb_symmetry():
    blocks = [
        basis_block.BasisBlock(
            center=np.array([0.0, 0.0, 1.0]),
            exponents=np.array([0.1, 0.2]),
            cartesian_powers=np.array([[0, 0, 0], [1, 0, 0]]),
            contraction_matrix=np.array([[0.6, 0.4], [0.3, 0.7]]),
            basis_transform=np.eye(2),
        ),
        basis_block.BasisBlock(
            center=np.array([1.0, 0.0, 1.0]),
            exponents=np.array([0.3, 0.4]),
            cartesian_powers=np.array([[0, 0, 0], [0, 1, 0]]),
            contraction_matrix=np.array([[0.5, 0.3], [0.1, 0.2]]),
            basis_transform=np.eye(2),
        ),
        basis_block.BasisBlock(
            center=np.array([0.0, 1.0, 0.0]),
            exponents=np.array([0.5, 0.6]),
            cartesian_powers=np.array([[0, 0, 0], [0, 0, 1]]),
            contraction_matrix=np.array([[0.4, 0.2], [0.3, 0.1]]),
            basis_transform=np.eye(2),
        ),
        basis_block.BasisBlock(
            center=np.array([1.0, 1.0, 0.0]),
            exponents=np.array([0.7, 0.8]),
            cartesian_powers=np.array([[0, 0, 0], [1, 0, 0]]),
            contraction_matrix=np.array([[0.9, 0.5], [0.6, 0.4]]),
            basis_transform=np.eye(2),
        ),
    ]

    permutations = [
        (1, 0, 2, 3),  #               (i,j)
        (0, 1, 3, 2),  #          (k,l)
        (1, 0, 3, 2),  #          (k,l)(i,j)
        (2, 3, 0, 1),  # (k,l,i,j)
        (2, 3, 1, 0),  # (k,l,i,j)     (i,j)
        (3, 2, 0, 1),  # (k,l,i,j)(k,l)
        (3, 2, 1, 0),  # (k,l,i,j)(k,l)(i,j)
    ]

    kernel = jit(
        functools.partial(
            operators.two_electron_matrix,
            operator=sf.integrals.two_electron,
        )
    )
    V = kernel(blocks[0], blocks[1], blocks[2], blocks[3])

    for sigma in permutations:
        V_perm = kernel(
            blocks[sigma[0]],
            blocks[sigma[1]],
            blocks[sigma[2]],
            blocks[sigma[3]],
        )
        np.testing.assert_allclose(
            V.transpose(sigma), V_perm, rtol=1e-8, atol=1e-8
        )
