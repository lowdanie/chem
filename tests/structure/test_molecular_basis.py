import numpy as np
from unittest import mock

from basis import contracted_gto
from structure import atom
from structure import molecule
from structure import molecular_basis


def _assert_atoms_equal(atom1: atom.Atom, atom2: atom.Atom):
    assert atom1.symbol == atom2.symbol
    assert atom1.number == atom2.number
    np.testing.assert_array_almost_equal(atom1.position, atom2.position)


@mock.patch("basis.bse_adapter.load")
def test_build_molecular_basis(mock_bse_load):
    mol = molecule.Molecule(
        atoms=[
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
    )
    basis_name = "sto-3g"

    contracted_gtos_H = [
        contracted_gto.ContractedGTO(
            primitive_type=contracted_gto.PrimitiveType.CARTESIAN,
            angular_momentum=(0,),
            exponents=np.array([1.0]),
            coefficients=np.array([[1.0]]),
        )
    ]

    contracted_gtos_O = [
        contracted_gto.ContractedGTO(
            primitive_type=contracted_gto.PrimitiveType.CARTESIAN,
            angular_momentum=(0,),
            exponents=np.array([2.0]),
            coefficients=np.array([[1.0]]),
        ),
        contracted_gto.ContractedGTO(
            primitive_type=contracted_gto.PrimitiveType.CARTESIAN,
            angular_momentum=(0,),
            exponents=np.array([3.0]),
            coefficients=np.array([[1.0]]),
        ),
    ]
    mock_bse_load.side_effect = [contracted_gtos_H, contracted_gtos_O]

    mol_basis = molecular_basis.build_molecular_basis(mol, basis_name)

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
