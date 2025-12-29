import dataclasses
import pytest

import jax
from jax import numpy as jnp
import numpy as np

from slaterform.hartree_fock import roothaan


@dataclasses.dataclass
class _OrthogonalizeBasisTestCase:
    # S = basis @ basis.T
    # shape: (l, n_basis)
    basis: jax.Array

    # Number of independent basis functions
    n_ind: int


@dataclasses.dataclass
class _SolveTestCase:
    # Fock matrix. Symmetric.
    # shape: (n_basis, n_basis)
    F: jax.Array

    # S = basis @ basis.T
    # shape: (l, n_basis)
    basis: jax.Array

    # Number of independent basis functions
    n_ind: int


# The basis is given by the columns of "basis".
@pytest.mark.parametrize(
    "case",
    [
        _OrthogonalizeBasisTestCase(
            basis=jnp.array(
                [
                    [1, 0],
                    [0, 1],
                ],
                dtype=jnp.float64,
            ),
            n_ind=2,
        ),
        _OrthogonalizeBasisTestCase(
            basis=jnp.array(
                [
                    [1, 0, 4],
                    [2, 3, 1],
                    [3, 1, 3],
                    [4, 5, 2],
                ],
                dtype=jnp.float64,
            ),
            n_ind=3,
        ),
        # linearly dependent basis
        _OrthogonalizeBasisTestCase(
            basis=jnp.array(
                [
                    [1, 2, 4],
                    [2, 4, 1],
                    [3, 6, 3],
                    [4, 8, 2],
                ],
                dtype=jnp.float64,
            ),
            n_ind=2,
        ),
    ],
)
def test_orthogonalize_basis(case):
    n_basis = case.basis.shape[1]

    S = case.basis.T @ case.basis  # shape (n_basis, n_basis)
    X = roothaan.orthogonalize_basis(S)  # shape (n_basis, n_basis)
    projection = np.diag(
        np.concatenate([np.zeros(n_basis - case.n_ind), np.ones(case.n_ind)])
    )

    # Check that X.T @ S @ X = projection
    np.testing.assert_allclose(
        X.T @ S @ X,
        projection,
        rtol=1e-7,
        atol=1e-7,
    )


@pytest.mark.parametrize(
    "case",
    [
        _SolveTestCase(
            F=jnp.array(
                [
                    [1, 0],
                    [0, 1],
                ],
                dtype=jnp.float64,
            ),
            basis=jnp.array(
                [
                    [1, 0],
                    [0, 1],
                ],
                dtype=jnp.float64,
            ),
            n_ind=2,
        ),
        _SolveTestCase(
            F=jnp.array(
                [
                    [1, 2, 3],
                    [2, 4, 5],
                    [3, 5, 6],
                ],
                dtype=np.float64,
            ),
            basis=jnp.array(
                [
                    [1, 0, 4],
                    [2, 3, 1],
                    [3, 1, 3],
                    [4, 5, 2],
                ],
                dtype=jnp.float64,
            ),
            n_ind=3,
        ),
        # linearly dependent basis
        _SolveTestCase(
            F=jnp.array(
                [
                    [1, 2, 3],
                    [2, 4, 6],
                    [3, 6, 6],
                ],
                dtype=jnp.float64,
            ),
            basis=jnp.array(
                [
                    [1, 2, 4],
                    [2, 4, 1],
                    [3, 6, 3],
                    [4, 8, 2],
                ],
                dtype=np.float64,
            ),
            n_ind=2,
        ),
    ],
)
def test_solve(case):
    S = case.basis.T @ case.basis  # shape (n_basis, n_basis)
    X = roothaan.orthogonalize_basis(S)  # shape (n_basis, n_basis)

    # orbital_energies: shape (n_basis,)
    # C_new: shape (n_basis, n_basis)
    orbital_energies, C_new = roothaan.solve(case.F, X)

    # Check that F C = S C E
    F_C = case.F @ C_new
    S_C_E = S @ C_new @ np.diag(orbital_energies)

    np.testing.assert_allclose(F_C, S_C_E, rtol=1e-7, atol=1e-7)
