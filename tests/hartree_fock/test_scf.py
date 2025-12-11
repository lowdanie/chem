import pytest

import numpy as np

from adapters import bse
from basis import contracted_gto
from structure import atom
from structure import molecule
from structure import molecular_basis
from hartree_fock import scf

_H2_MOLECULE = molecule.Molecule(
    atoms=[
        atom.Atom(
            symbol="H",
            number=1,
            position=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        ),
        atom.Atom(
            symbol="H",
            number=1,
            position=np.array([0.0, 0.0, 1.4], dtype=np.float64),
        ),
    ]
)

# The standard STO-3G basis set electronic energy for H2 at
# 1.4 Bohr bond length.
_EXPECTED_ENERGY_H2 = -1.8310  # Hartree

# Water molecule in Bohr units. The geometry is from pubchem.
_H20_MOLECULE = molecule.Molecule(
    atoms=[
        atom.Atom(
            symbol="O",
            number=8,
            position=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        ),
        atom.Atom(
            symbol="H",
            number=1,
            position=np.array(
                [0.52421003, 1.68733646, 0.48074633], dtype=np.float64
            ),
        ),
        atom.Atom(
            symbol="H",
            number=1,
            position=np.array(
                [1.14668581, -0.45032174, -1.35474466], dtype=np.float64
            ),
        ),
    ]
)


# The electronic energy for H2O with the STO-3G basis set.
# Computed with PySCF for reference.
# import numpy as np
# from pyscf import gto, scf
#
# mol = gto.M(
#     atom = '''
#     O 0.000000000  0.000000000 0.000000000
#     H 0.52421003 1.68733646 0.48074633
#     H 1.14668581, -0.45032174, -1.35474466
#     ''',
#     basis = 'sto-3g',
#     unit = 'Bohr',
#     symmetry = False
# )
#
# mf = scf.RHF(mol)
# mf.verbose = 4
# mf.kernel()
#
# print(f"Electronic Energy: {mf.e_tot - mol.energy_nuc():.8f} Ha")
_EXPECTED_ENERGY_H2O = -84.04881208  # Hartree


def _basis_fetcher(n: int) -> list[contracted_gto.ContractedGTO]:
    return bse.load("sto-3g", n)


def test_H2():
    mol_basis = molecular_basis.build(_H2_MOLECULE, _basis_fetcher)

    result = scf.solve(mol_basis)

    np.testing.assert_almost_equal(
        result.electronic_energy,
        _EXPECTED_ENERGY_H2,
        decimal=4,
    )

@pytest.mark.slow
def test_H2O():
    mol_basis = molecular_basis.build(_H20_MOLECULE, _basis_fetcher)

    result = scf.solve(mol_basis)

    np.testing.assert_almost_equal(
        result.electronic_energy,
        _EXPECTED_ENERGY_H2O,
        decimal=5,
    )
