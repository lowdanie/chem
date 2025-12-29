import jax
from jax import numpy as jnp


def orthogonalize_basis(
    S: jax.Array, linear_dep_threshold: float = 1e-8
) -> jax.Array:
    """
    Computes the orthogonalization matrix X = S^(-1/2).

    Handles linear dependencies by zeroizing eigenvectors with eigenvalues
    smaller than 'linear_dep_threshold'.

    Args:
        S: Overlap matrix (N, N)
        linear_dep_threshold: Eigenvalues smaller than this are zeroized.

    Returns:
        X: Transformation matrix of shape (N, N).
           Satisfies X.T @ S @ X = P, where P is a diagonal matrix with
           1s corresponding to valid eigenvalues and 0s for filtered ones.
           Note: Since eigh sorts ascendingly, P will typically look like
           diag(0, ..., 0, 1, ..., 1).
    """
    # Diagonalize S
    # S = U * s * U.T
    s, U = jnp.linalg.eigh(S)

    # Invert the square root of non-zero eigenvalues and set the rest to 0.
    mask = s > linear_dep_threshold
    safe_vals = jnp.where(mask, s, 1.0)
    inv_sqrt = 1.0 / jnp.sqrt(safe_vals)
    filtered_inv_sqrt = jnp.where(mask, inv_sqrt, 0.0)

    # Construct the canonical orthogonalization matrix
    # X = U * s^(-1/2)
    # We use broadcasting for the multiplication
    return U * filtered_inv_sqrt


def solve(F: jax.Array, X: jax.Array) -> tuple[jax.Array, jax.Array]:
    """
    Solves the Roothaan-Hall equations FC = SCE.

    Args:
        F: The Fock matrix in the atomic orbital basis. shape (N, N)
        X: The orthogonalization matrix such that X.T @ S @ X is equal to
           the projection matrix onto the orbitals. shape (N, N).

    Returns:
        orbital_energies: Array of shape (N,).
        coefficients: Matrix of shape (N, N).
    """
    # Transform Fock matrix to orthogonal basis: F' = X.T * F * X
    # shape: (M, M)
    F_prime = X.T @ F @ X

    # Diagonalize the transformed Fock matrix: F'C' = C'E
    # epsilon: Orbital energies
    # C_prime: Coefficients in the orthogonal basis. shape: (N, N)
    epsilon, C_prime = jnp.linalg.eigh(F_prime)

    # Transform coefficients back to original basis: C = X * C'
    # shape: (N, N)
    C = X @ C_prime

    return epsilon, C
