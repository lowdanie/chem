import dataclasses
import enum

import numpy as np


class PrimitiveType(enum.Enum):
    """The type of primitive Gaussian function."""

    CARTESIAN = 1
    SPHERICAL = 2


@dataclasses.dataclass
class ContractedGTO:
    """A contracted Gaussian-type orbital (Contracted GTO).

    This represents a family of contracted Gaussian basis functions centered at
    a given point in 3D space.

    A contracted Gaussian basis function is defined as a linear combination of
    normalized primitive Gaussian functions with the same center but different
    exponents:

    psi(r) = sum_{d=1}^K c_d * N(alpha_d, l, m) Y(r; l, m) e^(-alpha_d * |r - A|^2)

    where:
    1. A = (A_x, A_y, A_z) is the center
    2. r = (x, y, z) is the position coordinate
    3. alpha_d = exponents[d] for 0 <= d < K
    4. c_d = coefficients[s, d] for some 0 <= s < num_shells
    5. Y(r; l, m) is an (unnormalized) angular function with momentum
       l = angular_momentum[s]
       in either Cartesian or spherical representation, depending on primitive_type.
    6. N(alpha, l, m) is the inverse of the L^2 norm of
       Y(r; l, m) e^(-alpha_d * |r - A|^2)
    """

    primitive_type: PrimitiveType

    # The angular momentum for each shell in this contracted GTO.
    # shape (N_shell,)
    angular_momentum: tuple[int, ...]

    # The exponents for the Gaussian primitives in this contracted GTO.
    # shape (K,)
    exponents: np.ndarray

    # The contraction coefficients for each shell in this contracted GTO.
    # shape (N_shell, K)
    coefficients: np.ndarray
