import jax
from jax import numpy as jnp


def extract_tiles_2d(
    matrix: jax.Array,
    row_starts: jax.Array,
    col_starts: jax.Array,
    n_rows: int,
    n_cols: int,
) -> jax.Array:
    """Extracts a batch of 2D tiles from a matrix at specified offsets.

    Args:
        matrix: The matrix to extract from. Shape (M, N).
        row_starts: The row index for the start of each tile in the matrix.
            Shape (n_tiles,).
        col_starts: The col index for the start of each tile in the matrix.
            Shape (n_tiles,).
        n_rows: Number of rows in each tile.
        n_cols: Number of columns in each tile.

    Returns:
        The extracted tiles. Shape (n_tiles, n_rows, n_cols).
    """
    # shape: (n_tiles, n_rows, 1)
    # Add a dimension at the end for broadcasting against columns
    row_indices = (row_starts[:, None] + jnp.arange(n_rows))[:, :, None]

    # col_indices: (n_tiles, 1, n_cols)
    # Add a dimension in the middle for broadcasting against rows
    col_indices = (col_starts[:, None] + jnp.arange(n_cols))[:, None, :]

    return matrix[row_indices, col_indices]
