import itertools

import numpy as np

from slaterform.structure import molecule


def repulsion_energy(mol: molecule.Molecule) -> float:
    """Computes the nuclear repulsion energy for a given molecule."""
    energy = 0.0

    for atom1, atom2 in itertools.combinations(mol.atoms, 2):
        dist = np.linalg.norm(atom1.position - atom2.position)
        energy += (atom1.number * atom2.number) / dist

    return float(energy)
