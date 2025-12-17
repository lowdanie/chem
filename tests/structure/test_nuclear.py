import pytest

import numpy as np

from hflib.structure import atom
from hflib.structure import molecule
from hflib.structure import nuclear


@pytest.mark.parametrize(
    "mol, expected_energy",
    [
        (
            molecule.Molecule(
                atoms=[
                    atom.Atom(
                        symbol="O",
                        number=8,
                        position=np.array([0.0, 0.0, 0.0]),
                    ),
                    atom.Atom(
                        symbol="H",
                        number=1,
                        position=np.array([1.0, 0.0, 0.0]),
                    ),
                    atom.Atom(
                        symbol="Li",
                        number=3,
                        position=np.array([0.0, 2.0, 0.0]),
                    ),
                ]
            ),
            # 8 * 1 / 1.0 + 8 * 3 / 2.0 + 1 * 3 / np.sqrt(5)
            21.341640786499873,
        ),
        (
            molecule.Molecule(
                atoms=[
                    atom.Atom(
                        symbol="O",
                        number=8,
                        position=np.array([0.0, 0.0, 0.0]),
                    ),
                ]
            ),
            0.0,
        ),
    ],
)
def test_nuclear_repulsion_energy(
    mol: molecule.Molecule, expected_energy: float
):
    np.testing.assert_almost_equal(
        nuclear.repulsion_energy(mol), expected_energy, decimal=8
    )
