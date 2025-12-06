import dataclasses

import numpy as np


@dataclasses.dataclass
class ContractedGTO:
    """A contracted Gaussian-type orbital (Contracted GTO).

    This represents a family of contracted Gaussian basis functions centered at
    a given point in 3D space.

    A contracted Gaussian basis function is defined as a linear combination of
    normalized primitive Gaussian functions with the same center but different
    exponents:

    psi(r) = sum_{d=1}^K c_d * N(alpha_d, i, j, k) * G(r; A, alpha_d, i, j, k)

    where:
    1. A = (A_x, A_y, A_z) is the center,
    2. alpha_d = exponents[d] for 0 <= d < K
    3. c_d = coefficients[l, d] for some 0 <= l < num_shells
    4. i + j + k = angular_momentum[l]
    5. G(r; A, alpha, i, j, k) is the primitive Gaussian function defined by:
       (x - A_x)^i (y - A_y)^j (z - A_z)^k exp(-alpha * |r - A|^2)
    6. N(alpha, i, j, k) is the L^2 norm of G(r; A, alpha, i, j, k)
    """
    center: np.ndarray  # shape (3,)

    # The exponents for the Gaussian primitives in this contracted GTO.
    # shape (K,)
    exponents: np.ndarray

    # The angular momentum for each shell in this contracted GTO.
    # shape (N_shell,)
    angular_momentum: np.ndarray

    # The contraction coefficients for each shell in this contracted GTO.
    # shape (N_shell, K)
    coefficients: np.ndarray


@dataclasses.dataclass
class ContractedBasisBlock:
    """A block representation of a contracted Gaussian-type orbital basis.
    
    This flattens the angular momentum shells into their individual Cartesian
    components and pre-multiplies them by normalization constants. It also
    provides a basis transformation matrix from the Cartesian representation to
    the contracted basis function representation (Cartesian or spherical).
     
    Note that num_cartesian is equal to the total number of Cartesian basis
    functions across all shells, i.e.
    num_cartesian = sum_{l=0}^{num_shells - 1} (angular_momentum[l] + 2 choose 2)
    """
    center: np.ndarray  # shape (3,)

    # The common set of exponents for the Gaussian primitives in this block.
    # shape (K,)
    exponents: np.ndarray
    
    # The powers (i,j,k) for each Cartesian basis function in this block.
    # shape (N_cart, 3)
    cartesian_powers: np.ndarray

    # A map from the Gaussian primitives to their normalized contraction
    # coefficients for each Cartesian basis function in this block.
    # shape (N_cart, K)
    contraction_matrix: np.ndarray

    # A map from the Cartesian basis functions to the contracted basis functions
    # in this block.
    # shape (N_basis, N_cart)
    basis_transform: np.ndarray
