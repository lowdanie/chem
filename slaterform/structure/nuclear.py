import itertools
from collections.abc import Sequence

import jax
from jax import numpy as jnp

from slaterform.structure.atom import Atom


def repulsion_energy(atoms: Sequence[Atom]) -> jax.Array:
    """Computes the nuclear repulsion energy for a given molecule.

    Formula:
        E = sum_{i<j} (Z_i * Z_j) / |R_i - R_j|
    """
    # shape: (n_atoms, 3)
    positions = jnp.stack([a.position for a in atoms])
    # shape: (n_atoms,)
    charges = jnp.array([a.number for a in atoms])

    # Compute pairwise distances.
    # shape: (n_atoms, n_atoms, 3)
    deltas = positions[:, None, :] - positions[None, :, :]
    # shape: (n_atoms, n_atoms)
    diff_matrix = jnp.linalg.norm(deltas, axis=-1)

    # Iterate over pairs of atoms (i, j) with i < j to avoid double counting.
    idx_i, idx_j = jnp.triu_indices(len(atoms), k=1)

    r_ij = diff_matrix[idx_i, idx_j]
    z_i = charges[idx_i]
    z_j = charges[idx_j]

    return jnp.sum((z_i * z_j) / r_ij)
