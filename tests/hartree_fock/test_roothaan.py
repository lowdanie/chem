import dataclasses
import pytest

import numpy as np

from chem.hartree_fock import roothaan


@dataclasses.dataclass
class _OrthogonalizeBasisTestCase:
    # S = basis @ basis.T
    # shape: (l, n_basis)
    basis: np.ndarray

    # Number of independent basis functions
    n_ind: int


@dataclasses.dataclass
class _SolveTestCase:
    # Fock matrix. Symmetric.
    # shape: (n_basis, n_basis)
    F: np.ndarray

    # S = basis @ basis.T
    # shape: (l, n_basis)
    basis: np.ndarray

    # Number of independent basis functions
    n_ind: int


# The basis is given by the columns of "basis".
@pytest.mark.parametrize(
    "case",
    [
        _OrthogonalizeBasisTestCase(
            basis=np.array(
                [
                    [1, 0],
                    [0, 1],
                ],
                dtype=np.float64,
            ),
            n_ind=2,
        ),
        _OrthogonalizeBasisTestCase(
            basis=np.array(
                [
                    [1, 0, 4],
                    [2, 3, 1],
                    [3, 1, 3],
                    [4, 5, 2],
                ],
                dtype=np.float64,
            ),
            n_ind=3,
        ),
        # linearly dependent basis
        _OrthogonalizeBasisTestCase(
            basis=np.array(
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
def test_orthogonalize_basis(case):
    n_basis = case.basis.shape[1]

    S = np.matmul(case.basis.T, case.basis)  # shape (n_basis, n_basis)
    X = roothaan.orthogonalize_basis(S)  # shape (n_basis, k <= n_basis)
    assert X.shape == (n_basis, case.n_ind)

    # Check that X.T @ S @ X = I
    identity = np.eye(case.n_ind, dtype=np.float64)
    np.testing.assert_allclose(
        np.matmul(X.T, np.matmul(S, X)),
        identity,
        rtol=1e-7,
        atol=1e-7,
    )


@pytest.mark.parametrize(
    "case",
    [
        _SolveTestCase(
            F=np.array(
                [
                    [1, 0],
                    [0, 1],
                ],
                dtype=np.float64,
            ),
            basis=np.array(
                [
                    [1, 0],
                    [0, 1],
                ],
                dtype=np.float64,
            ),
            n_ind=2,
        ),
        _SolveTestCase(
            F=np.array(
                [
                    [1, 2, 3],
                    [2, 4, 5],
                    [3, 5, 6],
                ],
                dtype=np.float64,
            ),
            basis=np.array(
                [
                    [1, 0, 4],
                    [2, 3, 1],
                    [3, 1, 3],
                    [4, 5, 2],
                ],
                dtype=np.float64,
            ),
            n_ind=3,
        ),
        # linearly dependent basis
        _SolveTestCase(
            F=np.array(
                [
                    [1, 2, 3],
                    [2, 4, 6],
                    [3, 6, 6],
                ],
                dtype=np.float64,
            ),
            basis=np.array(
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
    S = np.matmul(case.basis.T, case.basis)  # shape (n_basis, n_ind)
    X = roothaan.orthogonalize_basis(S)  # shape (n_basis, n_ind)

    # orbital_energies: shape (n_ind,)
    # C_new: shape (n_basis, n_ind)
    orbital_energies, C_new = roothaan.solve(case.F, X)

    # Check that F C = S C E
    F_C = np.matmul(case.F, C_new)  # shape (n_basis, n_ind)
    S_C_E = np.matmul(
        S, np.matmul(C_new, np.diag(orbital_energies))
    )  # shape (n_basis, n_ind)

    np.testing.assert_allclose(F_C, S_C_E, rtol=1e-7, atol=1e-7)
