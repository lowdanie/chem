import jax
import jax.numpy as jnp

from slaterform import types


def broadcast_indices(*index_blocks: types.IntArray) -> tuple[jax.Array, ...]:
    """Reshapes columns of index blocks to broadcast against each other.

    Used to generate slicing indices for tensors constructed from multiple
    independent blocks (e.g. integral tensors).

    For example, given two index blocks I of shape (n1, 3) and J of shape
    (n2, 3), this function will return a tuple of 6 arrays:
    (I_x, I_y, I_z, J_x, J_y, J_z), where:
        - I_x,I_y,I_z have shape (n1, 1)
        - J_x,J_y,J_z have shape (1, n2)

        Furthermore, if T is a 6D tensor , we can generate a 2D tensor M
        with shape (n1,n2) defined by M[i,j] = T[I[i], J[j]] via:
        M = T[Ix, Iy, Iz, Jx, Jy, Jz]

    Args:
        *index_blocks: N arrays of shape (n_i, d).

    Returns:
        A tuple of d * len(index_blocks) arrays, where the columns
        of the d arrays for the i-th block have been reshaped to be orthogonal
        to the others.
    """
    indices = []
    n_blocks = len(index_blocks)

    for i, block in enumerate(index_blocks):
        # block shape: (N, D) -> e.g. (N, 3) for x,y,z
        # iterate over the columns (D)
        for col in jnp.asarray(block.T):
            # Create shape: (1, 1, ..., N, ..., 1)
            # where only the i-th position is -1 (N)
            shape = [1] * n_blocks
            shape[i] = -1
            indices.append(col.reshape(shape))

    return tuple(indices)


def flat_product(*arrays: jax.Array) -> tuple[jax.Array, ...]:
    """Generates the flattened cartesian product of input arrays.

    This should be equivalent to: zip(*itertools.product(*arrays))

    Example:
    a = [1, 2]
    b = [3, 4]
    flat_product(a, b) -> ([1, 1, 2, 2], [3, 4, 3, 4])

    Args:
        *arrays: N 1D arrays of shape (N_i,)
    Returns:
        A tuple of N arrays, each of shape (N_1 * ... * N_N,)
    """
    # Expand the dimensions of each array in the i-th dimension.
    ix = []
    for i, arr in enumerate(arrays):
        shape = [1] * len(arrays)
        shape[i] = -1
        ix.append(arr.reshape(shape))

    # Broadcast and flatten.
    grids = jnp.broadcast_arrays(*ix)
    return tuple(g.ravel() for g in grids)
