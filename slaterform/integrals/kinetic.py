import jax
import jax.numpy as jnp

from slaterform.integrals.gaussian import GaussianBasis3d
from slaterform.integrals.gaussian import gaussian_3d_to_1d
from slaterform.integrals.overlap import overlap_1d


def _kinetic_1d_from_overlap_1d(
    S: jax.Array,
    b: jax.Array,
) -> jax.Array:
    """Computes 1D kinetic energy integrals using recurrence on overlap integrals.

    Formula:
    T[i, j] = j(j-1)S[i, j-2] - 2b(2j+1)S[i, j] + 4b^2 S[i, j+2]

    Args:
        S: The 1D overlap matrix. Shape: (nrows, ncols) with ncols >= 3.
        b: The exponent of the 2nd Gaussian basis function.

    Returns:
        The kinetic energy matrix T.
        Shape: (nrows, ncols-2)
    """
    # The output T has shape (S.shape[0], S.shape[1]-2)
    nrows_t, ncols_t = S.shape[0], S.shape[1] - 2

    # Create the j index vector corresponding to the columns of T.
    # Shape: (1, ncols_t)
    j_vec = jnp.arange(ncols_t)[None, :]

    # Term 1: j(j-1) * S[i, j-2]
    # This term is only non-zero for j >= 2. In particular, this term
    # is only needed when S.shape[1] > 4.
    # We compute the values for j >= 2 and pad the first two columns with zeros.
    if ncols_t > 2:
        term1_vals = (j_vec[:, 2:] * (j_vec[:, 2:] - 1)) * S[:, :-4]
        term1 = jnp.pad(term1_vals, ((0, 0), (2, 0)))
    else:
        term1 = jnp.zeros((nrows_t, ncols_t), dtype=S.dtype)

    # Term 2: -2b(2j + 1) * S[i, j]
    term2 = -2 * b * (2 * j_vec + 1) * S[:, :-2]

    # Term 3: 4b^2 * S[i, j+2]
    # This term accesses S at indices 2, 3, ..., M-1.
    term3 = 4 * b**2 * S[:, 2:]

    return term1 + term2 + term3


def _kinetic_3d_from_overlap_1d(
    S_x: jax.Array,
    S_y: jax.Array,
    S_z: jax.Array,
    b: jax.Array,
) -> jax.Array:
    """Computes a 3d kinetic energy matrix from the 1d overlap matrices.

    Formula:
      T[ix, iy, iz, jx, jy, jz] =
          T_x[ix, jx] * S_y[iy, jy] * S_z[iz, jz] +
          S_x[ix, jx] * T_y[iy, jy] * S_z[iz, jz] +
          S_x[ix, jx] * S_y[iy, jy] * T_z[iz, jz]
    Args:
      S_x, S_y, S_z: The 1d overlap matrices.
        They all have the same shape (nrows, ncols) with ncols >= 3.
      b: The exponent of the 2nd Gaussian basis function.

    Returns:
      A 6-dimensional array T with shape
      (nrows, nrows, nrows, ncols-2, ncols-2, ncols-2)
    """
    T_x = _kinetic_1d_from_overlap_1d(S_x, b)
    T_y = _kinetic_1d_from_overlap_1d(S_y, b)
    T_z = _kinetic_1d_from_overlap_1d(S_z, b)

    # T_x[ix, jx] * S_y[iy, jy] * S_z[iz, jz]
    term_x = jnp.einsum("ad,be,cf->abcdef", T_x, S_y[:, :-2], S_z[:, :-2])

    # S_x[ix, jx] * T_y[iy, jy] * S_z[iz, jz]
    term_y = jnp.einsum("ad,be,cf->abcdef", S_x[:, :-2], T_y, S_z[:, :-2])

    # S_x[ix, jx] * S_y[iy, jy] * T_z[iz, jz]
    term_z = jnp.einsum("ad,be,cf->abcdef", S_x[:, :-2], S_y[:, :-2], T_z)

    return term_x + term_y + term_z


def kinetic_3d(g1: GaussianBasis3d, g2: GaussianBasis3d) -> jax.Array:
    """Calculates the matrix elements of the Laplacian operator.

    Formula:
        G1(x,y,z) = (x-Ax)^ix (y-Ay)^iy (z-Az)^iz e^(-a((x-Ax)^2+(y-Ay)^2+(z-Az)^2))
        G2(x,y,z) = (x-Bx)^jx (y-By)^jy (z-Bz)^jz e^(-b((x-Bx)^2+(y-By)^2+(z-Bz)^2))

        T[ix,iy,iz,jx,jy,jz] = integral
            G1(x1,y1,z1) * (d^2/dx^2 + d^2/dy^2 + d^2/dz^2)G2(x1,y1,z1)
            dx1 dy1 dz1 dx2 dy2 dz2

    Note: If you need the kinetic energy Hamiltonian, multiply the output of this
      function by -0.5.

    Args:
        g1: The first 3D Gaussian basis shell (center A, exponent alpha, max degree L_1).
        g2: The second 3D Gaussian basis shell (center B, exponent beta, max degree L_2).

    Returns:
        A 6-dimensional array S with shape: (L1+1, L1+1, L1+1, L2+1, L2+1, L2+1)
    """

    # Increase the degree of g2 by +2
    # We need this because the kinetic operator involves 2nd derivatives,
    # so we need overlaps of higher angular momentum to resolve it.
    g2_boosted = GaussianBasis3d(
        max_degree=g2.max_degree + 2, exponent=g2.exponent, center=g2.center
    )

    # Compute 1D Overlaps using the boosted g2
    # Shape: (d1+1, d2+3)
    g1x, g1y, g1z = [gaussian_3d_to_1d(g1, i) for i in range(3)]
    g2x, g2y, g2z = [gaussian_3d_to_1d(g2_boosted, i) for i in range(3)]

    S_x = overlap_1d(g1x, g2x)
    S_y = overlap_1d(g1y, g2y)
    S_z = overlap_1d(g1z, g2z)

    return _kinetic_3d_from_overlap_1d(S_x, S_y, S_z, jnp.asarray(g2.exponent))
