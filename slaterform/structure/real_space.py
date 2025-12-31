import numpy as np
import numpy.typing as npt

from slaterform.basis import real_space as basis_real_space
from slaterform.structure import batched_basis


def evaluate(
    basis: batched_basis.BatchedBasis, points: np.ndarray
) -> np.ndarray:
    """Evaluate the basis functions at specified points.

    Args:
        mol_basis: The molecular basis containing basis blocks.
        points: The points at which to evaluate the basis functions,
            shape (..., 3).
    Returns:
        The evaluated basis functions at each point, shape
        (..., mol.n_basis).
    """
    block_evals = []

    for block in basis.basis_blocks:
        block_eval = basis_real_space.evaluate(block, points)
        block_evals.append(block_eval)

    return np.concatenate(block_evals, axis=-1)
