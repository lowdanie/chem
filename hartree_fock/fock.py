import numpy as np

from basis import operators
from integrals import coulomb
from hartree_fock import quartet as quartet_lib
from structure import molecular_basis


def _scatter_integral_block(
    G: np.ndarray,
    P: np.ndarray,
    integral_block: np.ndarray,
    quartet: quartet_lib.Quartet,
    mol_basis: molecular_basis.MolecularBasis,
) -> None:
    slices = mol_basis.block_slices

    for sigma in quartet_lib.generate_permutations(quartet):
        # Permute the quartet and store the resulting indices.
        i, j, k, l = quartet_lib.apply_permutation(sigma, quartet)

        # Compute the two-electron integral block (ij|kl)
        ints = integral_block.transpose(sigma)

        # Add the Coulomb term.
        # G_ij += (ij|kl)*P_lk
        P_lk = P[slices[l], slices[k]]
        G[slices[i], slices[j]] += np.einsum("ijkl,lk->ij", ints, P_lk)

        # Add the exchange term.
        # G_il -= 0.5 * (ij|kl) * P_jk
        P_jk = P[slices[j], slices[k]]
        G[slices[i], slices[l]] -= 0.5 * np.einsum("ijkl,jk->il", ints, P_jk)


def two_electron_matrix(
    mol_basis: molecular_basis.MolecularBasis,
    P: np.ndarray,
) -> np.ndarray:
    """Computes the two electron part of the Fock matrix.

    The two-electron Fock matrix is given by:

    G_{ij} = sum_{kl} P_{kl} ( (ij|lk) - 0.5 (ik|lj) )

    where
    (ij|kl) = integral psi_i(r1) psi_j(r1) 1/|r1-r2| psi_k(r2) psi_l(r2) dr1 dr2

    Args:
        mol_basis: The molecular basis set.
        P: The closed shell density matrix of shape (n_basis, n_basis)
    Returns:
        A numpy array of shape (n_basis, n_basis)
    """
    n_basis = mol_basis.n_basis
    blocks = mol_basis.basis_blocks
    G = np.zeros((n_basis, n_basis), dtype=np.float64)

    # Iterate over canonical quartets of basis blocks.
    for quartet in quartet_lib.iter_canonical_quartets(len(blocks)):
        i, j, k, l = quartet

        # Compute the integral block (ij|kl)
        integral_block = operators.two_electron_matrix(
            blocks[i], blocks[j], blocks[k], blocks[l], coulomb.two_electron
        )

        # Scatter it to the full G matrix.
        _scatter_integral_block(G, P, integral_block, quartet, mol_basis)

    return G


def electronic_energy(
    H_core: np.ndarray, F: np.ndarray, P: np.ndarray
) -> np.float64:
    """Compute the electronic expectation energy from the Fock and density matrices.

    E = 0.5 * sum_{ij}P_ij(H_core_ji + F_ji)

    Args:
        H_core: The core Hamiltonian matrix. shape: (n_basis, n_basis)
        F: The core Fock matrix. shape: (n_basis, n_basis)
        P: The closed shell density matrix. shape: (n_basis, n_basis)
    """
    # Note that P is symmetric so we can use P_ij = P_ji
    return 0.5 * np.sum(P * (H_core + F))
