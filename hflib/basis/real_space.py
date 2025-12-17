import numpy as np
import numpy.typing as npt

from hflib.basis import basis_block


def evaluate(
    block: basis_block.BasisBlock, points: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Evaluate the basis functions in a basis block at given points.

    Args:
        block: The basis block to evaluate.
        points: The points at which to evaluate the basis functions,
            shape (..., 3).

    Returns:
        The evaluated basis functions at each point, shape
        (..., n_basis), where n_basis is the number of basis functions
        in the block.
    """
    # r - A
    # diff: (..., 3)
    diff = points - block.center

    # ||r - A||^2
    # dist_sq: (..., 1)
    dist_sq = np.sum(diff**2, axis=-1, keepdims=True)

    # e^{-alpha * r^2}
    # gaussians: (..., K)
    gaussians = np.exp(-block.exponents * dist_sq)

    # gaussians: (..., K)
    # contraction_matrix: (n_cart, K)
    # radial: (..., n_cart)
    # radial[..., i] = sum_k(gaussians[..., k] * contraction[i, k])
    radial = np.matmul(gaussians, block.contraction_matrix.T)

    # x^l * y^m * z^n
    # diff: (..., 3) -> (..., 1, 3)
    # powers: (n_cart, 3)
    # raised: (..., n_cart, 3)
    # angular: (..., n_cart)
    diff_expanded = diff[..., None, :]
    raised = np.power(diff_expanded, block.cartesian_powers)
    angular = np.prod(raised, axis=-1)

    # radial: (..., n_cart)
    # angular: (..., n_cart)
    # cart_eval: (..., n_cart)
    cart_eval = radial * angular

    # cart_eval: (..., n_cart)
    # basis_transform: (n_basis, n_cart)
    # result: (..., n_basis)
    return np.matmul(cart_eval, block.basis_transform.T)
