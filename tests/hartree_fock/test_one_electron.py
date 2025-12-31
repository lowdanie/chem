import pytest

from jax import jit
import numpy as np

import slaterform as sf
from slaterform.basis import operators

_TEST_MOLECULE = sf.Molecule(
    atoms=[
        sf.Atom(
            symbol="H",
            number=1,
            position=np.array([0.0, 0.0, 0.0]),
            shells=[
                sf.ContractedGTO(
                    primitive_type=sf.PrimitiveType.CARTESIAN,
                    angular_momentum=(0,),
                    exponents=np.array([0.1], dtype=np.float64),
                    coefficients=np.array([[1.0]], dtype=np.float64),
                )
            ],
        ),
        sf.Atom(
            symbol="O",
            number=8,
            position=np.array([0.0, 0.0, 1.0]),
            shells=[
                sf.ContractedGTO(
                    primitive_type=sf.PrimitiveType.CARTESIAN,
                    angular_momentum=(0, 1),
                    exponents=np.array([0.2], dtype=np.float64),
                    coefficients=np.array([[0.3], [0.4]], dtype=np.float64),
                )
            ],
        ),
    ]
)


def test_overlap_matrix():
    basis = sf.BatchedBasis.from_molecule(
        _TEST_MOLECULE, batch_size_1e=2, batch_size_2e=2
    )

    S_00 = operators.one_electron_matrix(
        basis.basis_blocks[0],
        basis.basis_blocks[0],
        sf.integrals.overlap_3d,
    )
    S_01 = operators.one_electron_matrix(
        basis.basis_blocks[0],
        basis.basis_blocks[1],
        sf.integrals.overlap_3d,
    )
    S_10 = S_01.T
    S_11 = operators.one_electron_matrix(
        basis.basis_blocks[1],
        basis.basis_blocks[1],
        sf.integrals.overlap_3d,
    )

    expected_S = np.vstack(
        [
            np.hstack([S_00, S_01]),
            np.hstack([S_10, S_11]),
        ]
    )

    S = jit(sf.hartree_fock.overlap_matrix)(basis)

    np.testing.assert_allclose(S, expected_S, rtol=1e-7, atol=1e-7)


def test_overlap_matrix_symmetric_pos_def():
    mol = sf.Molecule.from_geometry(_TEST_MOLECULE.atoms, basis_name="sto-3g")
    basis = sf.BatchedBasis.from_molecule(mol, batch_size_1e=2, batch_size_2e=2)
    S = jit(sf.hartree_fock.overlap_matrix)(basis)
    np.testing.assert_allclose(S, S.T, rtol=1e-7, atol=1e-7)

    # Check positive definiteness
    eigvals = np.linalg.eigvalsh(S)
    assert np.all(eigvals > 0.0)


def test_kinetic_matrix():
    basis = sf.BatchedBasis.from_molecule(
        _TEST_MOLECULE, batch_size_1e=2, batch_size_2e=2
    )

    T_00 = -0.5 * operators.one_electron_matrix(
        basis.basis_blocks[0],
        basis.basis_blocks[0],
        sf.integrals.kinetic_3d,
    )
    T_01 = -0.5 * operators.one_electron_matrix(
        basis.basis_blocks[0],
        basis.basis_blocks[1],
        sf.integrals.kinetic_3d,
    )
    T_10 = T_01.T
    T_11 = -0.5 * operators.one_electron_matrix(
        basis.basis_blocks[1],
        basis.basis_blocks[1],
        sf.integrals.kinetic_3d,
    )

    expected_T = np.vstack(
        [
            np.hstack([T_00, T_01]),
            np.hstack([T_10, T_11]),
        ]
    )

    T = jit(sf.hartree_fock.kinetic_matrix)(basis)
    np.testing.assert_allclose(T, expected_T, rtol=1e-7, atol=1e-7)


def test_kinetic_matrix_symmetric_pos_def():
    mol = sf.Molecule.from_geometry(
        atoms=_TEST_MOLECULE.atoms, basis_name="sto-3g"
    )
    basis = sf.BatchedBasis.from_molecule(mol, batch_size_1e=2, batch_size_2e=2)

    T = jit(sf.hartree_fock.kinetic_matrix)(basis)
    np.testing.assert_allclose(T, T.T, rtol=1e-7, atol=1e-7)

    # Check positive definiteness
    eigvals = np.linalg.eigvalsh(T)
    assert np.all(eigvals > 0.0)


def test_nuclear_attraction_matrix():
    def build_coulomb_operator(atom):
        return lambda g1, g2: sf.integrals.one_electron(g1, g2, atom.position)

    basis = sf.BatchedBasis.from_molecule(
        _TEST_MOLECULE, batch_size_1e=2, batch_size_2e=2
    )
    V0_00 = operators.one_electron_matrix(
        basis.basis_blocks[0],
        basis.basis_blocks[0],
        build_coulomb_operator(_TEST_MOLECULE.atoms[0]),
    )
    V1_00 = operators.one_electron_matrix(
        basis.basis_blocks[0],
        basis.basis_blocks[0],
        build_coulomb_operator(_TEST_MOLECULE.atoms[1]),
    )
    V_00 = -1 * V0_00 - 8 * V1_00

    V0_01 = operators.one_electron_matrix(
        basis.basis_blocks[0],
        basis.basis_blocks[1],
        build_coulomb_operator(_TEST_MOLECULE.atoms[0]),
    )
    V1_01 = operators.one_electron_matrix(
        basis.basis_blocks[0],
        basis.basis_blocks[1],
        build_coulomb_operator(_TEST_MOLECULE.atoms[1]),
    )
    V_01 = -1 * V0_01 - 8 * V1_01

    V_10 = V_01.T
    V0_11 = operators.one_electron_matrix(
        basis.basis_blocks[1],
        basis.basis_blocks[1],
        build_coulomb_operator(_TEST_MOLECULE.atoms[0]),
    )
    V1_11 = operators.one_electron_matrix(
        basis.basis_blocks[1],
        basis.basis_blocks[1],
        build_coulomb_operator(_TEST_MOLECULE.atoms[1]),
    )
    V_11 = -1 * V0_11 - 8 * V1_11
    expected_V = np.vstack(
        [
            np.hstack([V_00, V_01]),
            np.hstack([V_10, V_11]),
        ]
    )

    V = jit(sf.hartree_fock.nuclear_attraction_matrix)(basis)
    np.testing.assert_allclose(V, expected_V, rtol=1e-7, atol=1e-7)


def test_nuclear_attraction_matrix_symmetric_pos_def():
    mol = sf.Molecule.from_geometry(
        atoms=_TEST_MOLECULE.atoms, basis_name="sto-3g"
    )
    basis = sf.BatchedBasis.from_molecule(mol, batch_size_1e=2, batch_size_2e=2)

    V = jit(sf.hartree_fock.nuclear_attraction_matrix)(basis)
    np.testing.assert_allclose(V, V.T, rtol=1e-7, atol=1e-7)

    # Check negative definiteness
    eigvals = np.linalg.eigvalsh(V)
    assert np.all(eigvals < 0.0)
