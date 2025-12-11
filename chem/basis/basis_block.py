import dataclasses
from typing import Callable

import numpy as np

from chem.basis import contracted_gto
from chem.basis import cartesian


@dataclasses.dataclass
class BasisBlock:
    """A block representation of a contracted Gaussian-type orbital basis.

    This flattens the angular momentum shells into their individual Cartesian
    components and pre-multiplies them by normalization constants. It also
    provides a basis transformation matrix from the Cartesian representation to
    the contracted basis function representation (Cartesian or spherical).

    Note that N_cart is equal to the total number of Cartesian basis
    functions across all shells, i.e.
    N_cart = sum_{l=0}^{num_shells - 1} (angular_momentum[l] + 2 choose 2)
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

    # A map from the Cartesian basis functions to the angular basis functions
    # in this block (Cartesian or spherical).
    # shape (N_basis, N_cart)
    basis_transform: np.ndarray

    @property
    def n_cart(self) -> int:
        """The number of Cartesian basis functions in this block."""
        return self.cartesian_powers.shape[0]

    @property
    def n_basis(self) -> int:
        """The number of basis functions in this block."""
        return self.basis_transform.shape[0]


def _normalize_contraction_matrix(
    exponents: np.ndarray,
    contraction_matrix: np.ndarray,
    cartesian_powers: np.ndarray,
    max_degree: int,
) -> None:
    for i, a in enumerate(exponents):
        norms = cartesian.compute_normalization_constants(
            a, max_degree, cartesian_powers
        )
        contraction_matrix[:, i] *= norms


def build_basis_block(
    gto: contracted_gto.ContractedGTO, center: np.ndarray
) -> BasisBlock:
    """Builds a BasisBlock from a ContractedGTO at the given center."""
    if gto.primitive_type != contracted_gto.PrimitiveType.CARTESIAN:
        raise NotImplementedError(
            "Only Cartesian contracted GTOs are supported currently."
        )

    power_blocks = [
        cartesian.generate_cartesian_powers(l) for l in gto.angular_momentum
    ]
    power_block_sizes = [powers.shape[0] for powers in power_blocks]

    cartesian_powers = np.vstack(power_blocks)
    contraction_matrix = np.repeat(gto.coefficients, power_block_sizes, axis=0)
    _normalize_contraction_matrix(
        gto.exponents,
        contraction_matrix,
        cartesian_powers,
        max(gto.angular_momentum),
    )

    basis_transform = np.eye(cartesian_powers.shape[0], dtype=np.float64)

    return BasisBlock(
        center=center,
        exponents=gto.exponents,
        cartesian_powers=cartesian_powers,
        contraction_matrix=contraction_matrix,
        basis_transform=basis_transform,
    )
