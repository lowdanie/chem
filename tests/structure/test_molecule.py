from unittest import mock
import numpy as np

from tests.jax_utils import pytree_utils
import slaterform as sf

_DUMMY_SHELL_H = "Shell_H"
_DUMMY_SHELL_O = "Shell_O"


def _mock_bse_load_side_effect(basis_name, element):
    """Simulates fetching different shells for H (1) and O (8)."""
    if basis_name != "sto-3g":
        raise ValueError("Unexpected basis name")

    if element == 1:
        return [_DUMMY_SHELL_H]
    elif element == 8:
        return [_DUMMY_SHELL_O]
    else:
        raise ValueError(f"Unknown element: {element}")


def test_molecule_pytree():
    molecule = sf.Molecule(
        atoms=[
            sf.Atom(
                symbol="H",
                number=1,
                position=np.array([0.0, 0.0, 0.0]),
                shells=[
                    sf.ContractedGTO(
                        primitive_type=sf.PrimitiveType.CARTESIAN,
                        angular_momentum=(0,),
                        exponents=np.array([1.0]),
                        coefficients=np.array([2.0]),
                    ),
                ],
            ),
            sf.Atom(
                symbol="O",
                number=8,
                position=np.array([1.0, 0.0, 0.0]),
                shells=[
                    sf.ContractedGTO(
                        primitive_type=sf.PrimitiveType.CARTESIAN,
                        angular_momentum=(1,),
                        exponents=np.array([3.0]),
                        coefficients=np.array([4.0]),
                    ),
                ],
            ),
        ]
    )

    pytree_utils.assert_valid_pytree(molecule)


@mock.patch(
    "slaterform.structure.molecule.load_basis", side_effect=_mock_bse_load_side_effect
)
def test_from_geometry(mock_bse_load):
    geometry = [
        sf.Atom(
            symbol="H",
            number=1,
            position=np.array([0.0, 0.0, 0.0]),
            shells=[],
        ),
        sf.Atom(
            symbol="O",
            number=8,
            position=np.array([1.0, 0.0, 0.0]),
            shells=[],
        ),
    ]

    molecule = sf.Molecule.from_geometry(geometry, basis_name="sto-3g")

    assert len(molecule.atoms) == 2
    assert molecule.atoms[0].symbol == "H"
    assert molecule.atoms[0].number == 1
    np.testing.assert_array_equal(
        molecule.atoms[0].position, np.array([0.0, 0.0, 0.0])
    )
    assert molecule.atoms[0].shells == [_DUMMY_SHELL_H]
    assert molecule.atoms[1].symbol == "O"
    assert molecule.atoms[1].number == 8
    np.testing.assert_array_equal(
        molecule.atoms[1].position, np.array([1.0, 0.0, 0.0])
    )
    assert molecule.atoms[1].shells == [_DUMMY_SHELL_O]
