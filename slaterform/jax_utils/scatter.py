import jax
from jax import numpy as jnp


import jax
import jax.numpy as jnp


def add_tiles_2d(
    matrix: jax.Array,
    tiles: jax.Array,
    row_starts: jax.Array,
    col_starts: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    """Adds a batch of 2D tiles into a global matrix at specified offsets.

    Args:
        matrix: The matrix to update. Shape (M, N).
        tiles: The tiles to add. Shape (n_tiles, n_rows, n_cols).
        row_starts: The row index for the start of each tile in the matrix.
            Shape (batch,).
        col_starts: The col index for the start of each tile in the matrix.
        Shape (batch,).
        mask: Tiles where mask==0 are not added.
            Shape (batch,).

    Returns:
        The updated matrix.
    """
    n_rows = tiles.shape[1]
    n_cols = tiles.shape[2]

    # Apply the mask.
    # Zeros out values for padded batches so they sum to 0
    tiles = tiles * mask[:, None, None]

    # row_indices: (n_tiles, n_rows, 1)
    # Add a dimension at the end for broadcasting against columns
    row_indices = (row_starts[:, None] + jnp.arange(n_rows))[:, :, None]

    # col_indices: (n_tiles, 1, n_cols)
    # Add a dimension in the middle for broadcasting against rows
    col_indices = (col_starts[:, None] + jnp.arange(n_cols))[:, None, :]

    return matrix.at[row_indices, col_indices].add(tiles)
