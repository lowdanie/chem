import dataclasses

import numpy as np


@dataclasses.dataclass
class GaussianBasis1d:
    max_degree: int
    exponent: float
    center: float


@dataclasses.dataclass
class GaussianBasis3d:
    max_degree: int
    exponent: float
    center: np.ndarray  # shape (3,)


def gaussian_3d_to_1d(g: GaussianBasis3d, dim: int) -> GaussianBasis1d:
    return GaussianBasis1d(
        max_degree=g.max_degree, exponent=g.exponent, center=g.center[dim]
    )


def overlap_prefactor_1d(
    g1: GaussianBasis1d, g2: GaussianBasis1d
) -> float:
    mu = (g1.exponent * g2.exponent) / (g1.exponent + g2.exponent)
    diff = g1.center - g2.center
    return np.exp(-mu * np.square(diff))


def overlap_prefactor_3d(
    g1: GaussianBasis3d, g2: GaussianBasis3d
) -> float:
    mu = (g1.exponent * g2.exponent) / (g1.exponent + g2.exponent)
    diff = g1.center - g2.center
    return np.exp(-mu * np.dot(diff, diff))
