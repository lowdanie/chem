import dataclasses
from collections.abc import Sequence
from typing import Hashable

import jax
import jax.numpy as jnp
import numpy as np

from slaterform.basis import basis_block


@dataclasses.dataclass(frozen=True)
class BatchedBlocks:
    """A group of basis blocks batched for jax processing.

    The blocks in each stack all have the same static signature.
    """

    # block_stacks[i] is a stack of n_blocks[i] basis blocks.
    # Length: tuple_size
    block_stacks: tuple[basis_block.BasisBlock, ...]

    # Indices of the blocks starting indices in the global molecular basis.
    # global_block_starts[i] is an array of shape (n_blocks[i],)
    # Length: tuple_size
    global_block_starts: tuple[jax.Array, ...]

    # Batches of block tuple indices. The tuple indices index into the
    # block stacks.
    # Shape (n_batches, batch_size, tuple_size)
    tuple_indices: jax.Array

    # Mask indicating which entries in the batches are valid.
    # Shape (n_batches, batch_size)
    padding_mask: jax.Array


@dataclasses.dataclass(frozen=True, order=True)
class BlockSignature:
    aux_data: tuple[tuple[int, ...], ...]
    leaf_shapes: tuple[tuple[int, ...], ...]


_BlockGroupKey = tuple[BlockSignature, ...]


@dataclasses.dataclass(frozen=True)
class _BlockGroup:
    """A group of blocks with the same signature."""

    # blocks[i] is dict of BasisBlocks keyed with their global index.
    # Length: tuple_size
    blocks: tuple[dict[int, basis_block.BasisBlock], ...]

    # Each tuple has size tuple_size. The indices are into the global
    # list of blocks (i.e keys of blocks).
    tuple_indices: list[tuple[int, ...]]


def compute_block_signature(
    block: basis_block.BasisBlock,
) -> BlockSignature:
    """Computes a hashable static signature for a BasisBlock."""
    children, aux_data = block.tree_flatten()
    leaf_shapes = tuple(child.shape for child in children)
    return BlockSignature(aux_data=aux_data, leaf_shapes=leaf_shapes)


def _compute_group_key(
    blocks: Sequence[basis_block.BasisBlock],
    tuple_index: tuple[int, ...],
) -> _BlockGroupKey:
    return tuple(compute_block_signature(blocks[i]) for i in tuple_index)


def _init_block_group(
    tuple_size: int,
) -> _BlockGroup:
    return _BlockGroup(
        blocks=tuple(dict() for _ in range(tuple_size)),
        tuple_indices=[],
    )


def _update_block_group(
    block_group: _BlockGroup,
    blocks: Sequence[basis_block.BasisBlock],
    tuple_index: tuple[int, ...],
) -> None:
    block_group.tuple_indices.append(tuple_index)

    # Update the blocks if they are not already present.
    for i, block_idx in enumerate(tuple_index):
        if block_idx not in block_group.blocks[i]:
            block_group.blocks[i][block_idx] = blocks[block_idx]


def _generate_block_groups(
    blocks: Sequence[basis_block.BasisBlock],
    tuple_length: int,
    tuple_indices: Sequence[tuple[int, ...]],
) -> dict[_BlockGroupKey, _BlockGroup]:
    """Group the block tuples by their static signatures."""
    block_groups: dict[_BlockGroupKey, _BlockGroup] = {}

    for idx_tuple in tuple_indices:
        group_key = _compute_group_key(blocks, idx_tuple)
        if group_key not in block_groups:
            block_groups[group_key] = _init_block_group(tuple_length)
        _update_block_group(block_groups[group_key], blocks, idx_tuple)

    return block_groups


def _generate_block_stack(
    blocks: dict[int, basis_block.BasisBlock],
) -> tuple[basis_block.BasisBlock, np.ndarray]:
    """Generates a stack of basis blocks and their global indices."""
    # We're relying on the fact that dicts preserve insertion order.
    block_stack = list(blocks.values())
    global_indices = np.array(list(blocks.keys()))

    # Stack the arrays in the basis blocks
    block_stack = jax.tree.map(lambda *xs: jnp.stack(xs), *block_stack)
    return block_stack, global_indices


def _batch_tuple_indices(
    tuple_indices: np.ndarray, max_batch_size: int
) -> tuple[jax.Array, jax.Array]:
    """Batch tuple indices with padding."""
    n_tuples = tuple_indices.shape[0]
    tuple_size = tuple_indices.shape[1]

    batch_size = min(n_tuples, max_batch_size)
    remainder = n_tuples % batch_size
    pad_size = (batch_size - remainder) if remainder != 0 else 0

    # Pad the tuple indices to make n_tuples divisible by batch_size
    # shape: (a multiple of batch_size, tuple_size)
    padded_tuple_indices = np.pad(
        tuple_indices,
        ((0, pad_size), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    # Create a mask indicating the entries that are not padding.
    # shape: (a multiple of batch_size,)
    mask = np.concatenate([np.ones(n_tuples), np.zeros(pad_size)])

    batched_tuple_indices = padded_tuple_indices.reshape(
        -1, batch_size, tuple_size
    )
    batched_mask = mask.reshape(-1, batch_size)

    return jnp.array(batched_tuple_indices), jnp.array(batched_mask)


def _map_to_local_tuple_indices(
    global_tuple_indices: np.ndarray,
    global_block_indices: Sequence[np.ndarray],
) -> np.ndarray:
    """Maps global tuple indices to local tuple indices.

    Args:
        global_tuple_indices: shape (n_tuples, tuple_size)
        global_block_indices: Length tuple_size, each of shape (N_i,)
    """
    # shape: (n_tuples, tuple_size)
    local_tuple_indices = np.zeros_like(global_tuple_indices)
    for col, global_indices in enumerate(global_block_indices):
        max_global_idx = np.max(global_indices)
        lookup_table = np.zeros(max_global_idx + 1, dtype=np.int32)
        lookup_table[global_indices] = np.arange(
            len(global_indices), dtype=np.int32
        )
        local_tuple_indices[:, col] = lookup_table[global_tuple_indices[:, col]]

    return local_tuple_indices


def _batch_block_group(
    block_group: _BlockGroup, block_starts: np.ndarray, max_batch_size: int
) -> BatchedBlocks:
    # Generate tuple_size block stacks and global indices
    block_stacks = []
    global_block_indices = []
    global_block_starts = []

    for blocks_dict in block_group.blocks:
        block_stack, global_idx = _generate_block_stack(blocks_dict)
        block_stacks.append(block_stack)
        global_block_indices.append(global_idx)
        global_block_starts.append(block_starts[global_idx])

    # Convert to local tuple indices and batch them.
    global_tuple_indices = np.array(block_group.tuple_indices, dtype=np.int32)
    local_tuple_indices = _map_to_local_tuple_indices(
        global_tuple_indices, global_block_indices
    )
    batched_tuple_indices, padding_mask = _batch_tuple_indices(
        local_tuple_indices, max_batch_size
    )

    return BatchedBlocks(
        block_stacks=tuple(block_stacks),
        global_block_starts=tuple(jnp.array(g) for g in global_block_starts),
        tuple_indices=jnp.array(batched_tuple_indices),
        padding_mask=jnp.array(padding_mask),
    )


def _compute_block_starts(
    blocks: Sequence[basis_block.BasisBlock],
) -> np.ndarray:
    """Computes the starting indices of each block in the global basis."""
    if not blocks:
        return np.array([], dtype=np.int32)

    block_sizes = np.array([b.n_basis for b in blocks])
    return np.concatenate([np.array([0]), np.cumsum(block_sizes[:-1])])


def generate_batched_block_groups(
    blocks: Sequence[basis_block.BasisBlock],
    tuple_length: int,
    tuple_indices: Sequence[tuple[int, ...]],
    max_batch_size: int,
) -> Sequence[BatchedBlocks]:
    """Generates batched block groups from the given blocks and tuple indices.

    Note:
        The batched block tuples are grouped by their static signatures for efficient
        jax processing. If possible, sort the blocks to minimize the number of unique
        block tuple signatures before calling this function.
    Args:
        blocks: A sequence of BasisBlocks of length n_blocks.
        tuple_length: The length of each block tuple.
        tuple_indices: A sequence of tuples of indices of length
            n_tuples, each of length tuple_length. The indices in each tuple
            are between 0 and n_blocks-1.
        max_batch_size: The maximum batch size for batching the tuples.

    Returns:
        A sequence of BatchedBlocks, one for each unique block tuple signature.
        The tuples in the BatchedBlocks are equal to the input tuples of blocks.
    """
    block_starts = _compute_block_starts(blocks)
    block_groups = _generate_block_groups(blocks, tuple_length, tuple_indices)

    return [
        _batch_block_group(bg, block_starts, max_batch_size)
        for bg in block_groups.values()
    ]
