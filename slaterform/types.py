import jax
import numpy as np

Array = np.ndarray | jax.Array
Numeric = float | Array

IntArray = Array  # array of integers
Position3D = Array  # shape (3,)
Scalar = Numeric  # a single numeric value with no shape
