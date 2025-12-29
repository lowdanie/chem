import pytest

from jax import jit
import numpy as np

import slaterform as sf


@pytest.mark.parametrize(
    "mol, expected_energy",
    [
        (
            sf.Molecule(
                atoms=[
                    sf.Atom(
                        symbol="O",
                        number=8,
                        position=np.array([0.0, 0.0, 0.0]),
                    ),
                    sf.Atom(
                        symbol="H",
                        number=1,
                        position=np.array([1.0, 0.0, 0.0]),
                    ),
                    sf.Atom(
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
            sf.Molecule(
                atoms=[
                    sf.Atom(
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
def test_nuclear_repulsion_energy(mol: sf.Molecule, expected_energy: float):
    E = jit(sf.structure.nuclear_repulsion_energy)(mol)
    np.testing.assert_almost_equal(E, expected_energy, decimal=8)
