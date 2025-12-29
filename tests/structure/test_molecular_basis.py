import numpy as np

from jax import jit

import slaterform as sf


def _assert_atoms_equal(atom1: sf.Atom, atom2: sf.Atom):
    assert atom1.symbol == atom2.symbol
    assert atom1.number == atom2.number
    np.testing.assert_array_almost_equal(atom1.position, atom2.position)


def test_build_molecular_basis():
    mol = sf.Molecule(
        atoms=[
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
    )

    gtos = {
        1: [
            sf.ContractedGTO(
                primitive_type=sf.PrimitiveType.CARTESIAN,
                angular_momentum=(0,),
                exponents=np.array([1.0]),
                coefficients=np.array([[1.0]]),
            )
        ],
        8: [
            sf.ContractedGTO(
                primitive_type=sf.PrimitiveType.CARTESIAN,
                angular_momentum=(0,),
                exponents=np.array([2.0]),
                coefficients=np.array([[1.0]]),
            ),
            sf.ContractedGTO(
                primitive_type=sf.PrimitiveType.CARTESIAN,
                angular_momentum=(0, 1),
                exponents=np.array([3.0]),
                coefficients=np.array([[1.0], [0.5]]),
            ),
        ],
    }

    expected_num_basis = 6
    expected_num_electrons = 9

    mol_basis = jit(
        sf.structure.build_molecular_basis, static_argnames="basis_fetcher"
    )(mol, basis_fetcher=lambda n: gtos[n])

    assert mol_basis.n_basis == expected_num_basis
    assert mol_basis.n_electrons == expected_num_electrons

    assert len(mol_basis.atoms) == 2
    _assert_atoms_equal(mol_basis.atoms[0], mol.atoms[0])
    _assert_atoms_equal(mol_basis.atoms[1], mol.atoms[1])

    assert len(mol_basis.basis_blocks) == 3
    np.testing.assert_array_equal(
        mol_basis.basis_blocks[0].center, mol.atoms[0].position
    )
    np.testing.assert_array_equal(
        mol_basis.basis_blocks[1].center, mol.atoms[1].position
    )
    np.testing.assert_array_equal(
        mol_basis.basis_blocks[2].center, mol.atoms[1].position
    )


def test_molecule_property():
    atoms = [
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

    mol_basis = sf.MolecularBasis(
        atoms=atoms,
        basis_blocks=[],
    )

    mol = mol_basis.molecule
    assert len(mol.atoms) == 2
    _assert_atoms_equal(mol.atoms[0], atoms[0])
    _assert_atoms_equal(mol.atoms[1], atoms[1])
