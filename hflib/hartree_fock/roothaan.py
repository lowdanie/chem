import numpy as np
import scipy.linalg


def orthogonalize_basis(
    S: np.ndarray, linear_dep_threshold: float = 1e-8
) -> np.ndarray:
    """
    Computes the orthogonalization matrix X = S^(-1/2).

    Handles linear dependencies by discarding eigenvectors with eigenvalues
    smaller than 'linear_dep_threshold'.

    Args:
        S: Overlap matrix (N, N)
        linear_dep_threshold: Eigenvalues smaller than this are discarded.

    Returns:
        X: Transformation matrix of shape (N, M) where M <= N.
           Satisfies X.T @ S @ X = I (identity of size M).
    """
    # Diagonalize S
    # S = U * s * U.T
    vals, vecs = np.linalg.eigh(S)

    # Filter out small eigenvalues (Linear Dependency)
    mask = vals > linear_dep_threshold
    vals_good = vals[mask]  # shape: (M,)
    vecs_good = vecs[:, mask]  # shape: (N, M)

    # Construct the canonical orthogonalization matrix
    # X = U * s^(-1/2)
    # Shape: (N, M)
    # We use broadcasting for the multiplication
    X = vecs_good * (1.0 / np.sqrt(vals_good))

    return X


def solve(F: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Solves the Roothaan-Hall equations FC = SCE.

    Args:
        F: The Fock matrix in the atomic orbital basis. shape (N, N)
        X: The orthogonalization matrix such that X.T @ S @ X = I.
            shape (N, M) where M <= N.

    Returns:
        orbital_energies: Array of shape (M,) where M <= N.
        coefficients: Matrix of shape (N, M)
    """
    # Transform Fock matrix to orthogonal basis: F' = X.T * F * X
    # shape: (M, M)
    F_prime = np.matmul(X.T, np.matmul(F, X))

    # Diagonalize the transformed Fock matrix: F'C' = C'E
    # epsilon: Orbital energies
    # C_prime: Coefficients in the orthogonal basis. shape: (M, M)
    epsilon, C_prime = scipy.linalg.eigh(F_prime)

    # Transform coefficients back to original basis: C = X * C'
    # shape: (N, M)
    C = np.matmul(X, C_prime)

    return epsilon, C
