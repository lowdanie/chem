import slaterform as sf


def get_global_tuple_indices(
    batched_tree: sf.BatchedTreeTuples,
) -> list[tuple[int, ...]]:
    tuple_size = batched_tree.tuple_indices.shape[2]
    tuple_indices = batched_tree.tuple_indices.reshape(-1, tuple_size)
    padding_mask = batched_tree.padding_mask.reshape(-1)
    global_tuple_indices = []

    for idx_tuple, padding_mask in zip(tuple_indices, padding_mask):
        if padding_mask == 0:
            continue

        global_idx_tuple = tuple(
            int(batched_tree.global_tree_indices[i][idx_tuple[i]])
            for i in range(tuple_size)
        )
        global_tuple_indices.append(global_idx_tuple)

    return global_tuple_indices
