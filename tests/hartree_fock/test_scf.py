import pytest

import jax
from jax import jit
from jax import numpy as jnp
import numpy as np

import slaterform as sf
from slaterform.adapters import bse
from slaterform.basis import contracted_gto
from slaterform.structure import atom
from slaterform.structure import molecule
from slaterform.structure import molecular_basis
from slaterform.hartree_fock import scf

_H2_MOLECULE = sf.Molecule(
    atoms=[
        sf.Atom(
            symbol="H",
            number=1,
            position=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        ),
        sf.Atom(
            symbol="H",
            number=1,
            position=np.array([0.0, 0.0, 1.4], dtype=np.float64),
        ),
    ]
)

# The standard STO-3G basis set energies for H2 at
# 1.4 Bohr bond length.
_EXPECTED_ELECTRONIC_ENERGY_H2 = -1.8310  # Hartree
_EXPECTED_TOTAL_ENERGY_H2 = -1.1167  # Hartree

# Water molecule in Bohr units. The geometry is from pubchem.
_H20_MOLECULE = sf.Molecule(
    atoms=[
        sf.Atom(
            symbol="O",
            number=8,
            position=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        ),
        sf.Atom(
            symbol="H",
            number=1,
            position=np.array(
                [0.52421003, 1.68733646, 0.48074633], dtype=np.float64
            ),
        ),
        sf.Atom(
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
# print(f"Total Energy:      {mf.e_tot:.4f} Ha")
_EXPECTED_ELECTRONIC_ENERGY_H2O = -84.04881208  # Hartree
_EXPECTED_TOTAL_ENERGY_H20 = -74.96444758  # Hartree


def _basis_fetcher(n: int) -> list[sf.ContractedGTO]:
    return bse.load("sto-3g", n)


def test_H2():
    mol_basis = sf.structure.build_molecular_basis(_H2_MOLECULE, _basis_fetcher)
    batched_basis = sf.structure.batch_basis(mol_basis)
    result = scf.solve(batched_basis)

    np.testing.assert_almost_equal(
        result.electronic_energy,
        _EXPECTED_ELECTRONIC_ENERGY_H2,
        decimal=4,
    )
    np.testing.assert_almost_equal(
        result.total_energy,
        _EXPECTED_TOTAL_ENERGY_H2,
        decimal=4,
    )


@pytest.mark.slow
def test_H2O():
    mol_basis = sf.structure.build_molecular_basis(
        _H20_MOLECULE, _basis_fetcher
    )
    batched_basis = sf.structure.batch_basis(mol_basis)

    options = scf.Options(
        callback=lambda state: print(
            f"Iteration {state.iteration}: E = {state.electronic_energy:.8f} Ha"
        ),
    )
    result = scf.solve(batched_basis, options=options)

    np.testing.assert_almost_equal(
        result.electronic_energy,
        _EXPECTED_ELECTRONIC_ENERGY_H2O,
        decimal=5,
    )
    np.testing.assert_almost_equal(
        result.total_energy,
        _EXPECTED_TOTAL_ENERGY_H20,
        decimal=5,
    )
