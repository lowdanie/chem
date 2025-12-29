import itertools
import functools
from collections.abc import Sequence

from jax import jit
from jax import numpy as jnp
import numpy as np

import slaterform as sf
from slaterform.basis import operators

_TEST_ATOMS = [
    sf.Atom(
        symbol="H",
        number=1,
        position=np.array([0.0, 0.0, 0.0]),
    ),
    sf.Atom(
        symbol="O",
        number=8,
        position=np.array([0.0, 0.0, 1.0]),
    ),
]

_TEST_MOL_BASIS = sf.MolecularBasis(
    atoms=_TEST_ATOMS,
    basis_blocks=[
        sf.BasisBlock(
            center=np.array([0.0, 0.0, 0.0]),
            exponents=np.array([0.1], dtype=np.float64),
            cartesian_powers=np.array([[0, 0, 0]]),
            contraction_matrix=np.array([[1.0]], dtype=np.float64),
            basis_transform=np.eye(1, dtype=np.float64),
        ),
        sf.BasisBlock(
            center=np.array([0.0, 0.0, 1.0]),
            exponents=np.array([0.2], dtype=np.float64),
            cartesian_powers=np.array(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
            ),
            contraction_matrix=np.array(
                [[0.5], [0.4], [0.3], [0.2]],
                dtype=np.float64,
            ),
            basis_transform=np.eye(4, dtype=np.float64),
        ),
    ],
)

_TEST_P = np.array(
    [
        [1, 2, 3, 4, 5],
        [2, 6, 7, 8, 9],
        [3, 7, 1, 2, 3],
        [4, 8, 2, 4, 5],
        [5, 9, 3, 5, 6],
    ]
)

# Computed via _brute_force_two_electron_matrix(_TEST_MOL_BASIS, _TEST_P)
_EXPECTED_G = np.array(
    [
        [
            2.04806831e03,
            5.36657281e02,
            -8.60207706e01,
            -6.03069706e01,
            -1.26324841e02,
        ],
        [
            5.36657281e02,
            2.10668368e02,
            -2.83205385e01,
            -1.97124166e01,
            -6.40328089e00,
        ],
        [
            -8.60207706e01,
            -2.83205385e01,
            2.40075445e02,
            -3.11709391e00,
            1.62207725e00,
        ],
        [
            -6.03069706e01,
            -1.97124166e01,
            -3.11709391e00,
            1.33094254e02,
            8.26921197e-01,
        ],
        [
            -1.26324841e02,
            -6.40328089e00,
            1.62207725e00,
            8.26921197e-01,
            6.21964020e01,
        ],
    ]
)


def _build_slices(block_sizes: Sequence[int]) -> list[slice]:
    slices = []
    current = 0
    for size in block_sizes:
        slices.append(slice(current, current + size))
        current += size
    return slices


def _brute_force_two_electron_matrix(
    mol_basis: sf.MolecularBasis, P: np.ndarray
) -> np.ndarray:
    n_basis = mol_basis.n_basis
    blocks = mol_basis.basis_blocks
    slices = _build_slices([block.n_basis for block in blocks])

    V = np.zeros((n_basis, n_basis, n_basis, n_basis), dtype=np.float64)

    kernel = jit(
        functools.partial(
            operators.two_electron_matrix,
            operator=sf.integrals.two_electron,
        )
    )
    for i, j, k, l in itertools.product(range(len(blocks)), repeat=4):
        V_block = kernel(blocks[i], blocks[j], blocks[k], blocks[l])
        V[slices[i], slices[j], slices[k], slices[l]] = V_block

    # Compute G using the definition.
    # G_{ij} = sum_{kl} P_{kl} ( (ij|lk) - 0.5 (ik|lj) )
    G = np.zeros((n_basis, n_basis), dtype=np.float64)
    for i, j, k, l in itertools.product(range(n_basis), repeat=4):
        G[i, j] += P[k, l] * (V[i, j, l, k] - 0.5 * V[i, k, l, j])

    return G


def test_two_electron_matrix(
    mol_basis=_TEST_MOL_BASIS, P=_TEST_P, expected=_EXPECTED_G
):
    batched_basis = sf.structure.batch_basis(mol_basis)
    P_jax = jnp.asarray(P)

    G_result = sf.hartree_fock.two_electron_matrix_jax(batched_basis, P_jax)

    np.testing.assert_allclose(G_result, expected, rtol=1e-7, atol=1e-7)


def test_electronic_energy():
    H_core = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        dtype=np.float64,
    )

    F = np.array(
        [
            [1, 6, 7],
            [2, 5, 8],
            [3, 4, 9],
        ],
        dtype=np.float64,
    )

    P = np.array(
        [
            [9, 8, 7],
            [8, 6, 5],
            [7, 5, 4],
        ],
        dtype=np.float64,
    )

    # Compute the expectation energy directly.
    # E = 0.5 * sum_{ij}P_ij(H_core_ji + F_ji)
    E_expected = 0.0
    for i, j in itertools.product(range(3), repeat=2):
        E_expected += P[i, j] * (H_core[j, i] + F[j, i])
    E_expected *= 0.5

    E_result = jit(sf.hartree_fock.electronic_energy_jax)(H_core, F, P)

    np.testing.assert_almost_equal(E_result, E_expected, decimal=10)
