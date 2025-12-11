import pytest

import numpy as np

from adapters import bse
from basis import basis_block
from basis import contracted_gto
from basis import operators
from hartree_fock import one_electron
from integrals import overlap
from integrals import kinetic
from integrals import coulomb
from structure import atom
from structure import molecule
from structure import molecular_basis

_TEST_ATOMS = [
    atom.Atom(
        symbol="H",
        number=1,
        position=np.array([0.0, 0.0, 0.0]),
    ),
    atom.Atom(
        symbol="O",
        number=8,
        position=np.array([0.0, 0.0, 1.0]),
    ),
]

_TEST_MOL_BASIS = molecular_basis.MolecularBasis(
    atoms=_TEST_ATOMS,
    basis_blocks=[
        basis_block.BasisBlock(
            center=np.array([0.0, 0.0, 0.0]),
            exponents=np.array([0.1], dtype=np.float64),
            cartesian_powers=np.array([[0, 0, 0]]),
            contraction_matrix=np.array([[1.0]], dtype=np.float64),
            basis_transform=np.eye(1, dtype=np.float64),
        ),
        basis_block.BasisBlock(
            center=np.array([0.0, 0.0, 1.0]),
            exponents=np.array([0.2], dtype=np.float64),
            cartesian_powers=np.array(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
            ),
            contraction_matrix=np.array(
                [[0.5], [0.4], [0.3], [0.2]],
                dtype=np.float64,
            ),
            basis_transform=np.eye(4, dtype=np.float64),
        ),
    ],
    block_slices=[slice(0, 1), slice(1, 5)],
)


def _basis_fetcher(n: int) -> list[contracted_gto.ContractedGTO]:
    return bse.load("sto-3g", n)


def test_overlap_matrix():
    S_00 = operators.one_electron_matrix(
        _TEST_MOL_BASIS.basis_blocks[0],
        _TEST_MOL_BASIS.basis_blocks[0],
        overlap.overlap_3d,
    )
    S_01 = operators.one_electron_matrix(
        _TEST_MOL_BASIS.basis_blocks[0],
        _TEST_MOL_BASIS.basis_blocks[1],
        overlap.overlap_3d,
    )
    S_10 = S_01.T
    S_11 = operators.one_electron_matrix(
        _TEST_MOL_BASIS.basis_blocks[1],
        _TEST_MOL_BASIS.basis_blocks[1],
        overlap.overlap_3d,
    )

    expected_S = np.vstack(
        [
            np.hstack([S_00, S_01]),
            np.hstack([S_10, S_11]),
        ]
    )

    S = one_electron.overlap_matrix(_TEST_MOL_BASIS)
    np.testing.assert_allclose(S, expected_S, rtol=1e-7, atol=1e-7)


def test_overlap_matrix_symmetric_pos_def():
    mol = molecule.Molecule(atoms=_TEST_ATOMS)
    mol_basis = molecular_basis.build(mol, _basis_fetcher)

    S = one_electron.overlap_matrix(mol_basis)
    np.testing.assert_allclose(S, S.T, rtol=1e-7, atol=1e-7)

    # Check positive definiteness
    eigvals = np.linalg.eigvalsh(S)
    assert np.all(eigvals > 0.0)


def test_kinetic_matrix():
    T_00 = -0.5 * operators.one_electron_matrix(
        _TEST_MOL_BASIS.basis_blocks[0],
        _TEST_MOL_BASIS.basis_blocks[0],
        kinetic.kinetic_3d,
    )
    T_01 = -0.5 * operators.one_electron_matrix(
        _TEST_MOL_BASIS.basis_blocks[0],
        _TEST_MOL_BASIS.basis_blocks[1],
        kinetic.kinetic_3d,
    )
    T_10 = T_01.T
    T_11 = -0.5 * operators.one_electron_matrix(
        _TEST_MOL_BASIS.basis_blocks[1],
        _TEST_MOL_BASIS.basis_blocks[1],
        kinetic.kinetic_3d,
    )

    expected_T = np.vstack(
        [
            np.hstack([T_00, T_01]),
            np.hstack([T_10, T_11]),
        ]
    )

    T = one_electron.kinetic_matrix(_TEST_MOL_BASIS)
    np.testing.assert_allclose(T, expected_T, rtol=1e-7, atol=1e-7)


def test_kinetic_matrix_symmetric_pos_def():
    mol = molecule.Molecule(atoms=_TEST_ATOMS)
    mol_basis = molecular_basis.build(mol, _basis_fetcher)

    T = one_electron.kinetic_matrix(mol_basis)
    np.testing.assert_allclose(T, T.T, rtol=1e-7, atol=1e-7)

    # Check positive definiteness
    eigvals = np.linalg.eigvalsh(T)
    assert np.all(eigvals > 0.0)


def test_nuclear_attraction_matrix():
    def build_coulomb_operator(atom):
        return lambda g1, g2: coulomb.one_electron(g1, g2, atom.position)

    V0_00 = operators.one_electron_matrix(
        _TEST_MOL_BASIS.basis_blocks[0],
        _TEST_MOL_BASIS.basis_blocks[0],
        build_coulomb_operator(_TEST_ATOMS[0]),
    )
    V1_00 = operators.one_electron_matrix(
        _TEST_MOL_BASIS.basis_blocks[0],
        _TEST_MOL_BASIS.basis_blocks[0],
        build_coulomb_operator(_TEST_ATOMS[1]),
    )
    V_00 = -1 * V0_00 - 8 * V1_00

    V0_01 = operators.one_electron_matrix(
        _TEST_MOL_BASIS.basis_blocks[0],
        _TEST_MOL_BASIS.basis_blocks[1],
        build_coulomb_operator(_TEST_ATOMS[0]),
    )
    V1_01 = operators.one_electron_matrix(
        _TEST_MOL_BASIS.basis_blocks[0],
        _TEST_MOL_BASIS.basis_blocks[1],
        build_coulomb_operator(_TEST_ATOMS[1]),
    )
    V_01 = -1 * V0_01 - 8 * V1_01

    V_10 = V_01.T
    V0_11 = operators.one_electron_matrix(
        _TEST_MOL_BASIS.basis_blocks[1],
        _TEST_MOL_BASIS.basis_blocks[1],
        build_coulomb_operator(_TEST_ATOMS[0]),
    )
    V1_11 = operators.one_electron_matrix(
        _TEST_MOL_BASIS.basis_blocks[1],
        _TEST_MOL_BASIS.basis_blocks[1],
        build_coulomb_operator(_TEST_ATOMS[1]),
    )
    V_11 = -1 * V0_11 - 8 * V1_11
    expected_V = np.vstack(
        [
            np.hstack([V_00, V_01]),
            np.hstack([V_10, V_11]),
        ]
    )

    V = one_electron.nuclear_attraction_matrix(_TEST_MOL_BASIS)
    np.testing.assert_allclose(V, expected_V, rtol=1e-7, atol=1e-7)


def test_nuclear_attraction_matrix_symmetric_pos_def():
    mol = molecule.Molecule(atoms=_TEST_ATOMS)
    mol_basis = molecular_basis.build(mol, _basis_fetcher)

    V = one_electron.nuclear_attraction_matrix(mol_basis)
    np.testing.assert_allclose(V, V.T, rtol=1e-7, atol=1e-7)

    # Check negative definiteness
    eigvals = np.linalg.eigvalsh(V)
    assert np.all(eigvals < 0.0)
