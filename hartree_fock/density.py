import numpy as np


def closed_shell_matrix(C: np.ndarray, num_electrons: int) -> np.ndarray:
    """Compute the closed shell density matrix for the orbital coefficients.

    Args:
        C: The orbital coefficients. shape=(n_basis, n_basis)
        num_electrons: Compute the closed shell density using the first
            num_electrons//2 orbitals. num_electrons must be even.
    Returns:
        The closed shell density with shape (n_basis, n_basis)
    """
    if num_electrons % 2 != 0:
        raise ValueError(
            f"The number of electrons must be even. Got {num_electrons}"
        )

    C_occ = C[:, : num_electrons // 2]
    return 2 * np.matmul(C_occ, C_occ.T)
