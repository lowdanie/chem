import dataclasses
from typing import TypeAlias

import jax
from jax import numpy as jnp
import numpy as np

# Dynamic data
Array: TypeAlias = np.ndarray | jax.Array

# Static metadata
StaticArray: TypeAlias = np.ndarray


def promote_dataclass_fields(obj):
    """Converts all Array/StaticArray fields to jax/numpy arrays."""
    for field in dataclasses.fields(obj):
        value = getattr(obj, field.name)

        # Skip jax sentinels.
        if type(value) is object:
            continue

        if field.type == Array:
            setattr(obj, field.name, jnp.asarray(value))
        elif field.type == StaticArray:
            setattr(obj, field.name, np.asarray(value))
