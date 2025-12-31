import pytest

import jax
from jax import jit
from jax import numpy as jnp
import numpy as np

import slaterform as sf
from tests.jax_utils import pytree_utils

_H2_MOLECULE = sf.Molecule(
    atoms=[
        sf.Atom(
            symbol="H",
            number=1,
            position=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            shells=sf.adapters.bse.load("sto-3g", 1),
        ),
        sf.Atom(
            symbol="H",
            number=1,
            position=np.array([0.0, 0.0, 1.4], dtype=np.float64),
            shells=sf.adapters.bse.load("sto-3g", 1),
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
            shells=sf.adapters.bse.load("sto-3g", 8),
        ),
        sf.Atom(
            symbol="H",
            number=1,
            position=np.array(
                [0.52421003, 1.68733646, 0.48074633], dtype=np.float64
            ),
            shells=sf.adapters.bse.load("sto-3g", 1),
        ),
        sf.Atom(
            symbol="H",
            number=1,
            position=np.array(
                [1.14668581, -0.45032174, -1.35474466], dtype=np.float64
            ),
            shells=sf.adapters.bse.load("sto-3g", 1),
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


def test_options_pytree():
    options = sf.hartree_fock.scf.Options(
        max_iterations=50,
        convergence_threshold=1e-6,
        callback_interval=5,
        callback=lambda state: print("test"),
    )

    pytree_utils.assert_valid_pytree(options)


def test_options_zero_callback_intervals():
    with pytest.raises(ValueError):
        sf.hartree_fock.scf.Options(callback_interval=0)


def test_context_pytree():
    context = sf.hartree_fock.scf.Context(
        basis=sf.BatchedBasis.from_molecule(_H2_MOLECULE),
        nuclear_energy=jnp.asarray(1.0),
        S=jnp.ones((2, 2)),
        X=2 * jnp.ones((2, 2)),
        H_core=3 * jnp.ones((2, 2)),
    )

    pytree_utils.assert_valid_pytree(context)


def test_state_pytree():
    basis = sf.BatchedBasis.from_molecule(_H2_MOLECULE)
    context = sf.hartree_fock.scf.Context(
        basis=basis,
        nuclear_energy=jnp.asarray(1.0),
        S=jnp.ones((2, 2)),
        X=2 * jnp.ones((2, 2)),
        H_core=3 * jnp.ones((2, 2)),
    )
    state = sf.hartree_fock.scf.State(
        iteration=jnp.array(0, dtype=jnp.int32),
        context=context,
        C=4 * jnp.ones((2, 2)),
        P=5 * jnp.ones((2, 2)),
        F=6 * jnp.ones((2, 2)),
        electronic_energy=7 * jnp.asarray(1.0),
        total_energy=8 * jnp.asarray(1.0),
        orbital_energies=9 * jnp.ones((2,)),
        delta_P=10 * jnp.asarray(1.0),
    )

    pytree_utils.assert_valid_pytree(state)


def test_result_pytree():
    basis = sf.BatchedBasis.from_molecule(_H2_MOLECULE)
    result = sf.hartree_fock.scf.Result(
        converged=jnp.asarray(True),
        iterations=jnp.array(10, dtype=jnp.int32),
        electronic_energy=jnp.asarray(-1.0),
        nuclear_energy=jnp.asarray(1.0),
        total_energy=jnp.asarray(0.0),
        basis=basis,
        orbital_energies=jnp.ones((2,)),
        orbitals=jnp.ones((2, 2)),
        density=jnp.ones((2, 2)),
    )

    pytree_utils.assert_valid_pytree(result)


def test_H2():
    basis = sf.BatchedBasis.from_molecule(_H2_MOLECULE)
    result = jit(sf.hartree_fock.scf.solve)(basis)

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


def test_H2_from_molecule():
    result = jit(sf.hartree_fock.scf.solve)(_H2_MOLECULE)

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
    basis = sf.BatchedBasis.from_molecule(_H20_MOLECULE)
    options = sf.hartree_fock.scf.Options(
        callback_interval=5,
        callback=lambda state: print(
            f"Iteration {state.iteration}: E = {state.electronic_energy:.8f} Ha"
        ),
    )
    result = jit(sf.hartree_fock.scf.solve)(basis, options=options)

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
