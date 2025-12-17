import numpy as np
import numpy.typing as npt

from hflib.basis import real_space as basis_real_space
from hflib.structure import molecular_basis


def evaluate(
    mol_basis: molecular_basis.MolecularBasis, points: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Evaluate the basis functions at specified points.

    Args:
        mol_basis: The molecular basis containing basis blocks.
        points: The points at which to evaluate the basis functions,
            shape (..., 3).
    Returns:
        The evaluated basis functions at each point, shape
        (..., mol.n_basis).
    """
    block_evals: list[npt.NDArray[np.float64]] = []

    for block in mol_basis.basis_blocks:
        block_eval = basis_real_space.evaluate(block, points)
        block_evals.append(block_eval)

    return np.concatenate(block_evals, axis=-1)
