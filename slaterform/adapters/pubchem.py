from collections.abc import Sequence
import numpy as np
import pubchempy as pcp

from slaterform.structure.atom import Atom
import slaterform.structure.units as units


def _load_atom(atom_data: pcp.Atom) -> Atom:
    position = (
        np.array([atom_data.x, atom_data.y, atom_data.z], dtype=np.float64)
        * units.ANGSTROM_TO_BOHR
    )

    return Atom(
        symbol=atom_data.element,
        number=atom_data.number,
        position=position,
        shells=[],
    )


def load_geometry(compound: pcp.Compound) -> Sequence[Atom]:
    if compound.coordinate_type != "3d":
        raise ValueError("Compound must have 3D coordinates.")

    return [_load_atom(atom_data) for atom_data in compound.atoms]
