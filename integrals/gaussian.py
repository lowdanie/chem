import dataclasses

import numpy as np

@dataclasses.dataclass
class CartesianGaussian1d:
    max_degree: int
    exponent: float
    center: float


def overlap_prefactor_1d(
    g1: CartesianGaussian1d, g2: CartesianGaussian1d
) -> float:
    q = (g1.exponent * g2.exponent) / (g1.exponent + g2.exponent)
    diff = g1.center - g2.center
    return np.exp(-q * np.square(diff))
