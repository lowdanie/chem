import numpy as np
import pubchempy as pcp

from slaterform.structure import molecule
from slaterform.structure import atom
from slaterform.structure import units


def _load_atom(atom_data: pcp.Atom) -> atom.Atom:
    position = (
        np.array([atom_data.x, atom_data.y, atom_data.z], dtype=np.float64)
        * units.ANGSTROM_TO_BOHR
    )

    return atom.Atom(
        symbol=atom_data.element,
        number=atom_data.number,
        position=position,
    )


def load_molecule(compound: pcp.Compound) -> molecule.Molecule:
    if compound.coordinate_type != "3d":
        raise ValueError("Compound must have 3D coordinates.")

    atoms = [_load_atom(atom_data) for atom_data in compound.atoms]
    return molecule.Molecule(atoms=atoms)
