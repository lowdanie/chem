import numpy as np
import scipy.linalg


def orthogonalize_basis(S: np.ndarray) -> np.ndarray:
    """
    Computes the symmetric orthogonalization matrix X = S^(-1/2).

    Args:
        S: The overlap matrix of shape (N, N).
    Returns:
        A numpy array of shape (N, N) satisfying X.T @ S @ X = I.
    """
    # Diagonalize S
    vals, vecs = np.linalg.eigh(S)

    # 2. Compute Lambda^(-1/2)
    vals_inv_sqrt = 1.0 / np.sqrt(vals)

    # Construct X = U @ Lambda^(-1/2) @ U.T
    # The diagonal matrix Lambda^(-1/2) is applied via broadcasting.
    X = vecs * vals_inv_sqrt @ vecs.T

    return X


def solve(F: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Solves the Roothaan-Hall equations FC = SCE.

    Args:
        F: The Fock matrix in the atomic orbital basis.
        X: The orthogonalization matrix (S^-1/2).

    Returns:
        orbital_energies: Array of shape (N,)
        coefficients: Matrix of shape (N, N) where columns are eigenvectors.
    """
    # Transform Fock matrix to orthogonal basis: F' = X.T * F * X
    F_prime = X.T @ F @ X

    # Diagonalize the transformed Fock matrix: F'C' = C'E
    # epsilon: Orbital energies
    # C_prime: Coefficients in the orthogonal basis
    epsilon, C_prime = scipy.linalg.eigh(F_prime)

    # Transform coefficients back to original basis: C = X * C'
    C = X @ C_prime

    return epsilon, C
