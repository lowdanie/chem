import dataclasses
import pytest

import numpy as np

from slaterform.basis import cartesian


@dataclasses.dataclass
class _GenerateCartesianPowersTestCase:
    l: int
    expected: np.ndarray


@dataclasses.dataclass
class _ComputeNormalizationConstantsTestCase:
    exponent: float
    max_degree: int
    cartesian_powers: np.ndarray
    expected: np.ndarray


@pytest.mark.parametrize(
    "case",
    [
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
    ],
)
def test_generate_cartesian_powers(case):
    powers = cartesian.generate_cartesian_powers(case.l)

    np.testing.assert_array_equal(powers, case.expected)


def test_generate_cartesian_powers_invalid_l():
    with pytest.raises(ValueError):
        cartesian.generate_cartesian_powers(2)


# Values computed using sympy for cartesian powers (i,j,k):
# import numpy as np
# import sympy as sp
#
# x, y, z = sp.symbols('x,y,z')
# a=0.1
#
# integrand = (x**i * y**j * z**k * sp.exp(-a *(x**2 + y**2 + z**2)))**2
# res = sp.integrate(integrand, (x,-sp.oo,sp.oo), (y, -sp.oo, sp.oo), (z, -sp.oo, sp.oo))
# norm_sq = float(res.evalf())
# inv_norm = 1 / np.sqrt(norm_sq)
# print(f"[i,j,k]: ", inv_norm)
@pytest.mark.parametrize(
    "case",
    [
        _ComputeNormalizationConstantsTestCase(
            exponent=0.1,
            max_degree=1,
            cartesian_powers=np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ]
            ),
            expected=np.array(
                [
                    0.12673895,
                    0.08015675,
                    0.08015675,
                    0.08015675,
                ]
            ),
        ),
    ],
)
def test_compute_normalization_constants(case):
    norms = cartesian.compute_normalization_constants(
        case.max_degree,
        case.cartesian_powers,
        np.array(case.exponent),
    )

    np.testing.assert_allclose(norms, case.expected, atol=1e-7)
