import dataclasses
import pytest
import unittest
from unittest import mock

import numpy as np

from hflib.structure import atom
from hflib.structure import molecule
from hflib.adapters import pubchem
from hflib.structure import units


@dataclasses.dataclass
class _AtomData:
    element: str
    number: int
    x: float
    y: float
    z: float


@dataclasses.dataclass
class _LoadMoleculeTestCase:
    coordinate_type: str
    atom_data: list[_AtomData]
    expected_molecule: molecule.Molecule


def _assert_atoms_equal(atom1: atom.Atom, atom2: atom.Atom):
    assert atom1.symbol == atom2.symbol
    assert atom1.number == atom2.number
    np.testing.assert_array_almost_equal(atom1.position, atom2.position)


@pytest.mark.parametrize(
    "case",
    [
        _LoadMoleculeTestCase(
            coordinate_type="3d",
            atom_data=[
                _AtomData(element="H", number=1, x=0.1, y=0.0, z=0.0),
                _AtomData(element="O", number=8, x=0.0, y=0.2, z=0.0),
                _AtomData(element="H", number=1, x=0.0, y=0.0, z=0.3),
            ],
            expected_molecule=molecule.Molecule(
                atoms=[
                    atom.Atom(
                        symbol="H",
                        number=1,
                        position=np.array([0.1, 0.0, 0.0])
                        * units.ANGSTROM_TO_BOHR,
                    ),
                    atom.Atom(
                        symbol="O",
                        number=8,
                        position=np.array([0.0, 0.2, 0.0])
                        * units.ANGSTROM_TO_BOHR,
                    ),
                    atom.Atom(
                        symbol="H",
                        number=1,
                        position=np.array([0.0, 0.0, 0.3])
                        * units.ANGSTROM_TO_BOHR,
                    ),
                ]
            ),
        ),
    ],
)
def test_load_molecule(case):
    mock_compound = mock.MagicMock()
    mock_compound.coordinate_type = case.coordinate_type
    mock_compound.atoms = [
        mock.MagicMock(
            element=atom_data.element,
            number=atom_data.number,
            x=atom_data.x,
            y=atom_data.y,
            z=atom_data.z,
        )
        for atom_data in case.atom_data
    ]

    loaded_molecule = pubchem.load_molecule(mock_compound)
    assert len(loaded_molecule.atoms) == len(case.expected_molecule.atoms)
    for loaded_atom, expected_atom in zip(
        loaded_molecule.atoms, case.expected_molecule.atoms
    ):
        _assert_atoms_equal(loaded_atom, expected_atom)


def test_load_molecule_invalid_coordinate_type():
    mock_compound = mock.MagicMock()
    mock_compound.coordinate_type = "2d"  # Invalid coordinate type

    with pytest.raises(ValueError, match="Compound must have 3D coordinates."):
        pubchem.load_molecule(mock_compound)
