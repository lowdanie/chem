import dataclasses
import pytest

import jax.numpy as jnp
import numpy as np

import slaterform as sf


@dataclasses.dataclass
class _TestCase3dSparse:
    g1: sf.GaussianBasis3d
    g2: sf.GaussianBasis3d
    expected_shape: tuple[int, int, int, int, int, int]
    coords: np.ndarray  # shape (n, 6)
    expected: np.ndarray  # shape (n,)


# The expected values were computed using sympy.
# For a given array of coord of shape (6,), we compute the integral:
# import sympy as sp
# G1 = (x + 2)**coord[0] * (y - 0)**coord[1] * (z - 1)**coord[2] *\
#      sp.exp(-0.5 * ((x + 2)**2 + (y - 0)**2 + (z - 1)**2))
# G2 = (x - 1)**coord[3] * (y - 2)**coord[4] * (z + 1)**coord[5] *\
#      sp.exp(-0.2 * ((x - 1)**2 + (y - 2)**2 + (z + 1)**2))
# integrand = G1 * (sp.diff(G2, x, 2) +\
#                   sp.diff(G2, y, 2) +\
#                   sp.diff(G2, z, 2))
# res = sp.integrate(integrand,
#                    (x, -sp.oo, sp.oo),
#                    (y, -sp.oo, sp.oo),
#                    (z, -sp.oo, sp.oo))
# expected = float(res.evalf())
_TEST_CASES_3D = [
    _TestCase3dSparse(
        g1=sf.GaussianBasis3d(
            max_degree=2,
            exponent=0.5,
            center=np.array([-2.0, 0.0, 1.0]),
        ),
        g2=sf.GaussianBasis3d(
            max_degree=1,
            exponent=0.2,
            center=np.array([1.0, 2.0, -1.0]),
        ),
        expected_shape=(3, 3, 3, 2, 2, 2),
        coords=np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [2, 1, 0, 1, 0, 1],
                [2, 2, 2, 1, 1, 1],
            ]
        ),
        expected=np.array(
            [
                0.44477446568451884,
                2.0360720605318163,
                -0.7693172854848808,
            ]
        ),
    )
]


@pytest.mark.parametrize("case", _TEST_CASES_3D)
def test_kinetic_3d(case):
    T = sf.integrals.kinetic_3d_jax(case.g1, case.g2)
    assert T.shape == case.expected_shape

    indices = tuple(case.coords.T)
    actual_values = T[indices]

    np.testing.assert_allclose(actual_values, case.expected, atol=1e-7)
