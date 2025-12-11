import numpy as np

import itertools

from integrals import overlap
from integrals import kinetic
from integrals import coulomb
from structure import molecular_basis
from basis import operators


def one_electron_matrix(
    mol_basis: molecular_basis.MolecularBasis,
    operator: operators.OneElectronOperator,
) -> np.ndarray:
    """Computes the one-electron matrix for a given molecular basis.

    Returns:
        A numpy array of shape (N, N) where N=mol_basis.n_basis
    """
    output = np.empty((mol_basis.n_basis, mol_basis.n_basis), dtype=np.float64)

    for i, j in itertools.combinations_with_replacement(
        range(len(mol_basis.basis_blocks)), 2
    ):
        block1, slice1 = mol_basis.basis_blocks[i], mol_basis.block_slices[i]
        block2, slice2 = mol_basis.basis_blocks[j], mol_basis.block_slices[j]

        block_matrix = operators.one_electron_matrix(block1, block2, operator)

        output[slice1, slice2] = block_matrix

        if i != j:
            output[slice2, slice1] = block_matrix.T

    return output


def overlap_matrix(
    mol_basis: molecular_basis.MolecularBasis,
) -> np.ndarray:
    """Computes the overlap matrix S

    Returns:
        A numpy array of shape (N, N) where N=mol_basis.n_basis
    """
    S = one_electron_matrix(
        mol_basis,
        overlap.overlap_3d,
    )

    return S


def nuclear_attraction_matrix(
    mol_basis: molecular_basis.MolecularBasis,
) -> np.ndarray:
    """Computes the nuclear attraction matrix V

    Returns:
        A numpy array of shape (N, N) where N=mol_basis.n_basis
    """
    V = np.zeros((mol_basis.n_basis, mol_basis.n_basis), dtype=np.float64)
    for atom in mol_basis.atoms:
        V -= atom.number * one_electron_matrix(
            mol_basis,
            lambda g1, g2: coulomb.one_electron(g1, g2, atom.position),
        )

    return V


def kinetic_matrix(
    mol_basis: molecular_basis.MolecularBasis,
) -> np.ndarray:
    """Computes the kinetic energy matrix T

    Returns:
        A numpy array of shape (N, N) where N=mol_basis.n_basis
    """
    return -0.5 * one_electron_matrix(
        mol_basis,
        kinetic.kinetic_3d,
    )


def core_hamiltonian_matrix(
    mol_basis: molecular_basis.MolecularBasis,
) -> np.ndarray:
    """Computes the core Hamiltonian matrix H = T + V

    Returns:
        A numpy array of shape (N, N) where N=mol_basis.n_basis
    """
    T = kinetic_matrix(mol_basis)
    V = nuclear_attraction_matrix(mol_basis)

    return T + V
