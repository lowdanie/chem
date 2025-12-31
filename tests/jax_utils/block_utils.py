import slaterform as sf


def get_global_tuple_batches(
    batched_tree: sf.BatchedTreeTuples,
) -> list[list[tuple[int, ...]]]:
    tuple_size = batched_tree.tuple_indices.shape[2]
    global_tuple_batches = []

    for tuple_batch, padding_mask_batch in zip(
        batched_tree.tuple_indices, batched_tree.padding_mask
    ):
        batch = []
        for idx_tuple, padding_mask in zip(tuple_batch, padding_mask_batch):
            if padding_mask == 0:
                continue

            global_idx_tuple = tuple(
                int(batched_tree.global_tree_indices[i][idx_tuple[i]])
                for i in range(tuple_size)
            )
            batch.append(global_idx_tuple)

        global_tuple_batches.append(batch)

    return global_tuple_batches
