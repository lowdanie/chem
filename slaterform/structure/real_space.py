from collections.abc import Sequence

import numpy as np

from slaterform.basis.real_space import evaluate as evaluate_basis
from slaterform.basis.basis_block import BasisBlock


def evaluate(
    basis_blocks: Sequence[BasisBlock], points: np.ndarray
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

    for block in basis_blocks:
        block_eval = evaluate_basis(block, points)
        block_evals.append(block_eval)

    return np.concatenate(block_evals, axis=-1)
