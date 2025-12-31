import jax
import numpy as np


def assert_pytrees_equal(obj1, obj2):
    """
    Asserts that two JAX Pytree objects are structurally identical and have equal leaf values.
    """
    treedef1 = jax.tree_util.tree_structure(obj1)
    treedef2 = jax.tree_util.tree_structure(obj2)

    assert treedef1 == treedef2, "Pytree structures do not match!"

    leaves1 = jax.tree_util.tree_leaves(obj1)
    leaves2 = jax.tree_util.tree_leaves(obj2)

    for l1, l2 in zip(leaves1, leaves2):
        np.testing.assert_array_equal(
            l1, l2, err_msg="Pytree leaf values do not match!"
        )


def assert_valid_pytree(obj):
    """
    Verifies that a JAX Pytree object can be flattened, unflattened,
    and passed through a JIT boundary correctly.
    """
    # Test flatten/unflatten
    leaves, treedef = jax.tree_util.tree_flatten(obj)
    obj_reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
    assert_pytrees_equal(obj, obj_reconstructed)

    # Check that the object can be passed through a JIT boundary
    obj_jitted = jax.jit(lambda x: x)(obj)
    assert_pytrees_equal(obj, obj_jitted)
