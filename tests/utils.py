import jax
import jax.numpy as jnp
import numpy as np


def assert_no_nan(array: np.ndarray | jax.Array):
    assert not np.any(np.isnan(array)), "Array contains NaN values."
