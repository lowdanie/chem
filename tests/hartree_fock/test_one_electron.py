import pytest

from jax import jit
import numpy as np

import slaterform as sf
from slaterform.basis import operators

_TEST_ATOMS = [
    sf.Atom(
        symbol="H",
        number=1,
        position=np.array([0.0, 0.0, 0.0]),
    ),
    sf.Atom(
        symbol="O",
        number=8,
        position=np.array([0.0, 0.0, 1.0]),
    ),
]

_TEST_MOL_BASIS = sf.MolecularBasis(
    atoms=_TEST_ATOMS,
    basis_blocks=[
        sf.BasisBlock(
            center=np.array([0.0, 0.0, 0.0]),
            exponents=np.array([0.1], dtype=np.float64),
            cartesian_powers=np.array([[0, 0, 0]]),
            contraction_matrix=np.array([[1.0]], dtype=np.float64),
            basis_transform=np.eye(1, dtype=np.float64),
        ),
        sf.BasisBlock(
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
)


def _basis_fetcher(n: int) -> list[sf.ContractedGTO]:
    return sf.adapters.bse.load("sto-3g", n)


def test_overlap_matrix():
    S_00 = operators.one_electron_matrix(
        _TEST_MOL_BASIS.basis_blocks[0],
        _TEST_MOL_BASIS.basis_blocks[0],
        sf.integrals.overlap_3d,
    )
    S_01 = operators.one_electron_matrix(
        _TEST_MOL_BASIS.basis_blocks[0],
        _TEST_MOL_BASIS.basis_blocks[1],
        sf.integrals.overlap_3d,
    )
    S_10 = S_01.T
    S_11 = operators.one_electron_matrix(
        _TEST_MOL_BASIS.basis_blocks[1],
        _TEST_MOL_BASIS.basis_blocks[1],
        sf.integrals.overlap_3d,
    )

    expected_S = np.vstack(
        [
            np.hstack([S_00, S_01]),
            np.hstack([S_10, S_11]),
        ]
    )

    basis = sf.structure.batch_basis(
        _TEST_MOL_BASIS, batch_size_1e=2, batch_size_2e=2
    )
    S = jit(sf.hartree_fock.overlap_matrix_jax)(basis)

    np.testing.assert_allclose(S, expected_S, rtol=1e-7, atol=1e-7)


def test_overlap_matrix_symmetric_pos_def():
    mol = sf.Molecule(atoms=_TEST_ATOMS)
    mol_basis = sf.structure.build_molecular_basis(mol, _basis_fetcher)
    batched_basis = sf.structure.batch_basis(
        mol_basis, batch_size_1e=2, batch_size_2e=2
    )
    S = jit(sf.hartree_fock.overlap_matrix_jax)(batched_basis)
    np.testing.assert_allclose(S, S.T, rtol=1e-7, atol=1e-7)

    # Check positive definiteness
    eigvals = np.linalg.eigvalsh(S)
    assert np.all(eigvals > 0.0)


def test_kinetic_matrix():
    T_00 = -0.5 * operators.one_electron_matrix(
        _TEST_MOL_BASIS.basis_blocks[0],
        _TEST_MOL_BASIS.basis_blocks[0],
        sf.integrals.kinetic_3d,
    )
    T_01 = -0.5 * operators.one_electron_matrix(
        _TEST_MOL_BASIS.basis_blocks[0],
        _TEST_MOL_BASIS.basis_blocks[1],
        sf.integrals.kinetic_3d,
    )
    T_10 = T_01.T
    T_11 = -0.5 * operators.one_electron_matrix(
        _TEST_MOL_BASIS.basis_blocks[1],
        _TEST_MOL_BASIS.basis_blocks[1],
        sf.integrals.kinetic_3d,
    )

    expected_T = np.vstack(
        [
            np.hstack([T_00, T_01]),
            np.hstack([T_10, T_11]),
        ]
    )

    basis = sf.structure.batch_basis(
        _TEST_MOL_BASIS, batch_size_1e=2, batch_size_2e=2
    )
    T = jit(sf.hartree_fock.kinetic_matrix_jax)(basis)
    np.testing.assert_allclose(T, expected_T, rtol=1e-7, atol=1e-7)


def test_kinetic_matrix_symmetric_pos_def():
    mol = sf.Molecule(atoms=_TEST_ATOMS)
    mol_basis = sf.structure.build_molecular_basis(mol, _basis_fetcher)
    batched_basis = sf.structure.batch_basis(
        _TEST_MOL_BASIS, batch_size_1e=2, batch_size_2e=2
    )
    T = jit(sf.hartree_fock.kinetic_matrix_jax)(batched_basis)
    np.testing.assert_allclose(T, T.T, rtol=1e-7, atol=1e-7)

    # Check positive definiteness
    eigvals = np.linalg.eigvalsh(T)
    assert np.all(eigvals > 0.0)


def test_nuclear_attraction_matrix():
    def build_coulomb_operator(atom):
        return lambda g1, g2: sf.integrals.one_electron(g1, g2, atom.position)

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

    basis = sf.structure.batch_basis(
        _TEST_MOL_BASIS, batch_size_1e=2, batch_size_2e=2
    )
    V = jit(sf.hartree_fock.nuclear_attraction_matrix_jax)(basis)
    np.testing.assert_allclose(V, expected_V, rtol=1e-7, atol=1e-7)


def test_nuclear_attraction_matrix_symmetric_pos_def():
    mol = sf.Molecule(atoms=_TEST_ATOMS)
    mol_basis = sf.structure.build_molecular_basis(mol, _basis_fetcher)
    batched_basis = sf.structure.batch_basis(
        _TEST_MOL_BASIS, batch_size_1e=2, batch_size_2e=2
    )
    V = jit(sf.hartree_fock.nuclear_attraction_matrix_jax)(batched_basis)
    np.testing.assert_allclose(V, V.T, rtol=1e-7, atol=1e-7)

    # Check negative definiteness
    eigvals = np.linalg.eigvalsh(V)
    assert np.all(eigvals < 0.0)
