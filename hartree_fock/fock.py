import dataclasses

import numpy as np

from integrals import gaussian


@dataclasses.dataclass
class MonomialBasis:
    """A monomial basis in variables x, y, z up to a given degree.

    The number of monomials in 3 variables up to degree d is:
    (d + 3 choose 3)
    """

    max_degree: int

    # Each array has shape (((d + 3) choose 3),)
    # The monomials in the basis are:
    # x^indices[0][i] * y^indices[1][i] * z^indices[2][i]
    # for 0 <= i < (d + 3 choose 3)
    indices: tuple[np.ndarray, np.ndarray, np.ndarray]

    def size(self) -> int:
        return self.indices[0].shape[0]

def compress(
    coeffs: np.ndarray, basis: MonomialBasis
) -> np.ndarray:
    """Compress a 3D array of monomial coefficients to a 1D array.

    Args:
        coeffs: A 3D array of shape (d+1, d+1, d+1) where d = basis.max_degree.
        basis: The MonomialBasis defining which monomials to keep.
    Returns:
        A 1D array of shape (d + 3 choose 3) containing the coefficients
        of the monomials in the basis. The output coeffs_1d satisfies:
        coeffs_1d[i] = coeffs[indices[0][i], indices[1][i], indices[2][i]]
    """
    return coeffs[basis.indices]

def decompress(
    coeffs_1d: np.ndarray, basis: MonomialBasis
) -> np.ndarray:
    """Decompress a 1D array of monomial coefficients to a 3D array.

    Args:
        coeffs_1d: A 1D array of shape (d + 3 choose 3) where d = basis.max_degree.
        basis: The MonomialBasis defining which monomials to keep.
    Returns:
        A 3D array of shape (d+1, d+1, d+1) containing the coefficients
        of all monomials up to degree d. The output coeffs satisfies:
        coeffs[indices[0][i], indices[1][i], indices[2][i]] = coeffs_1d[i]
    """
    d = basis.max_degree
    coeffs = np.zeros((d + 1, d + 1, d + 1), dtype=coeffs_1d.dtype)
    coeffs[basis.indices] = coeffs_1d
    return coeffs


# Suppose we are using a basis with N GaussianBasis3d object
# G1, G2, ..., GN.
# The overlap array S is a two dimensional array of shape (N, N).
# For each pair of basis functions (Gi, Gj), S[i,j] is a 6D array of shape
# (Gi.max_degree + 1,)*3 + (Gj.max_degree + 1,)*3.
#
OverlapArray = np.ndarray


def compute_overlap_array(
    basis: list[gaussian.GaussianBasis3d],
) -> OverlapArray:
    N = len(basis)
    S = np.empty((N, N), dtype=object)
