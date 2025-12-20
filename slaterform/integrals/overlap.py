from typing import NamedTuple
import functools

import jax
from jax import jit
import jax.numpy as jnp
import numpy as np

from slaterform.integrals import gaussian


def _base_case(
    g1: gaussian.GaussianBasis1d, g2: gaussian.GaussianBasis1d
) -> jax.Array:
    p = g1.exponent + g2.exponent
    K = gaussian.overlap_prefactor_1d_jax(g1, g2)
    return jnp.sqrt(np.pi / p) * K


class _VerticalTransferParams(NamedTuple):
    PA: jax.Array  # P - A
    inv_2p: jax.Array  # 1/(2p)


def _vertical_transfer_step(
    params: _VerticalTransferParams,
    carry: tuple[jax.Array, jax.Array],
    i: jax.Array,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    """Computes the next recurrence step for the vertical transfer.

    Formula: S[i] = (P - A) * S[i-1] + ((i-1)/2p) * S[i-2]

    Args:
      params: The static coefficients (P-A, 1/2p).
      carry: A tuple of scalars (S[i-1], S[i-2]) representing the previous two vertical steps.
      i: The current vertical index (recurrence depth).

    Returns:
      A tuple (new_carry, output):
        new_carry: (S[i], S[i-1]) for the next step.
        output: S[i] to be stacked into the result array.
    """
    PA, inv_2p = params

    # S[i-1], S[i-2]
    s_im1, s_im2 = carry

    # S[i] = (P - A) * S[i-1] + ((i-1)/2p) * S[i-2]
    term1 = PA * s_im1
    term2 = (i - 1) * inv_2p * s_im2
    s_i = term1 + term2

    return (s_i, s_im1), s_i


def _vertical_transfer(
    s_base: jax.Array,
    g1: gaussian.GaussianBasis1d,
    g2: gaussian.GaussianBasis1d,
) -> jax.Array:
    """Computes the vertical recurrence column S[:, 0].

    Generates S[i, 0] for i in [0, total_degree], where total_degree =
    g1.max_degree + g2.max_degree.

    Args:
      s_base: The base integral S[0,0].

    Returns:
      An array S[:,0] of shape (g1.max_degree + g2.max_degree + 1,).
    """
    p = jnp.asarray(g1.exponent + g2.exponent)
    P = jnp.asarray((g1.exponent * g1.center + g2.exponent * g2.center) / p)
    A = jnp.asarray(g1.center)
    params = _VerticalTransferParams(PA=P - A, inv_2p=1 / (2 * p))
    step_fn = functools.partial(_vertical_transfer_step, params)

    init_carry = (s_base, jnp.zeros_like(s_base))
    indices = jnp.arange(1, g1.max_degree + g2.max_degree + 1)

    _, s_rest = jax.lax.scan(step_fn, init_carry, indices)
    return jnp.concatenate([jnp.array([s_base]), s_rest])


class _HorizontalTransferParams(NamedTuple):
    AB: jax.Array  # A - B


def _horizontal_transfer_step(
    params: _HorizontalTransferParams, s_jm1: jax.Array, j: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Computes the next column (j) from the previous column (j-1).

    Formula: S[i, j] = (A - B) * S[i, j-1] + S[i+1, j-1]

    Args:
      params: The static coefficient (A-B).
      s_jm1: The previous column S[:, j-1] of shape (N,).
             Assume that S[i,j-1] is valid for 0 <= i < N - (j-1).
      j: The current horizontal index (recurrence depth). Unused.

    Returns:
      A tuple (new_carry, output):
        new_carry: The current column S[:, j] of shape (N,).
                   S[i,j] is valid for 0 <= i < N - j.
        output: Same as new_carry.
    """
    # S[i+1,j-1]
    s_jm1_up = jnp.roll(s_jm1, shift=-1, axis=0)

    # S[i,j] = AB * S[i, j-1] + S[i+1, j-1]
    s_j = params.AB * s_jm1 + s_jm1_up
    return s_j, s_j


def _horizontal_transfer(
    s_j0: jax.Array, g1: gaussian.GaussianBasis1d, g2: gaussian.GaussianBasis1d
) -> jax.Array:
    """Applies horizontal transfer to generate columns 1 through M.

    Args:
      s_j0: The 0-th column S[:, 0] of shape (N,).

    Returns:
      The full overlap matrix S of shape (N, g2.max_degree + 1).
      S[i,j] is valid for 0 <= i < N - j.
    """
    AB = jnp.asarray(g1.center - g2.center)
    params = _HorizontalTransferParams(AB=AB)
    step_fn = functools.partial(_horizontal_transfer_step, params)

    indices = jnp.arange(1, g2.max_degree + 1)

    _, s_rest = jax.lax.scan(step_fn, s_j0, indices)
    return jnp.hstack([s_j0[:, None], s_rest.T])


@jit
def overlap_1d_jax(
    g1: gaussian.GaussianBasis1d, g2: gaussian.GaussianBasis1d
) -> jax.Array:
    """Computes the 1D overlap matrix between two Gaussians.

    Args:
      g1,g2: The Gaussians.

    Returns:
      An array S of shape (g1.max_degree + 1, g2.max_degree + 1).
    """
    # Base case: S[0,0]
    s_00 = _base_case(g1, g2)

    # Vertical transfer: S[:, 0]
    # s_col0 has shape (g1.max_degree + g2.max_degree + 1,)
    s_col0 = _vertical_transfer(s_00, g1, g2)

    # Horizontal Transfer: Compute remaining columns S[:, 1:]
    # S has shape (g1.max_degree + g2.max_degree + 1, g2.max_degree + 1)
    S = _horizontal_transfer(s_col0, g1, g2)

    return S[: g1.max_degree + 1, :]


def _overlap_3d_from_1d(
    S_x: jax.Array, S_y: jax.Array, S_z: jax.Array
) -> jax.Array:
    """Compute the 3d overlap matrices from the 1d overlap matrices.

    Formula:
      S[ix, iy, iz, jx, jy, jz] = S_x[ix, jx] * S_y[iy, jy] * S_z[iz, jz]

    Args:
      S_x, S_y, S_z: The 1d overlap matrices.
    Returns:
      An array S of shape
      (S_x.shape[0], S_y.shape[0], S_z.shape[0],
       S_x.shape[1], S_y.shape[1], S_z.shape[1])
    """
    return jnp.einsum("ad,be,cf->abcdef", S_x, S_y, S_z)


@jit
def overlap_3d_jax(
    g1: gaussian.GaussianBasis3d, g2: gaussian.GaussianBasis3d
) -> jax.Array:
    """Computes the overlap integrals between two 3D Gaussian basis shells.

    Formula:
        G1(x,y,z) = (x-Ax)^ix (y-Ay)^iy (z-Az)^iz e^(-a((x-Ax)^2+(y-Ay)^2+(z-Az)^2))
        G2(x,y,z) = (x-Bx)^jx (y-By)^jy (z-Bz)^jz e^(-b((x-Bx)^2+(y-By)^2+(z-Bz)^2))

        S[ix,iy,iz,jx,jy,jz] =
            integral G1(x1,y1,z1) * G2(x1,y1,z1) dx1 dy1 dz1 dx2 dy2 dz2

    Args:
        g1: The first 3D Gaussian basis shell (center A, exponent alpha, max degree L_1).
        g2: The second 3D Gaussian basis shell (center B, exponent beta, max degree L_2).

    Returns:
        A 6-dimensional array S with shape: (L1+1, L1+1, L1+1, L2+1, L2+1, L2+1)
    """
    S_x = overlap_1d_jax(
        gaussian.gaussian_3d_to_1d_jax(g1, 0),
        gaussian.gaussian_3d_to_1d_jax(g2, 0),
    )
    S_y = overlap_1d_jax(
        gaussian.gaussian_3d_to_1d_jax(g1, 1),
        gaussian.gaussian_3d_to_1d_jax(g2, 1),
    )
    S_z = overlap_1d_jax(
        gaussian.gaussian_3d_to_1d_jax(g1, 2),
        gaussian.gaussian_3d_to_1d_jax(g2, 2),
    )

    return _overlap_3d_from_1d(S_x, S_y, S_z)


def overlap_1d(
    g1: gaussian.GaussianBasis1d, g2: gaussian.GaussianBasis1d
) -> np.ndarray:
    return np.array(overlap_1d_jax(g1, g2))


def overlap_3d(
    g1: gaussian.GaussianBasis3d, g2: gaussian.GaussianBasis3d
) -> np.ndarray:
    return np.array(overlap_3d_jax(g1, g2))
