import itertools

import numpy as np

from hflib.basis import basis_block
from hflib.basis import operators
from hflib.integrals import coulomb
from hflib.structure import atom
from hflib.structure import molecular_basis
from hflib.hartree_fock import fock

_TEST_ATOMS = [
    atom.Atom(
        symbol="H",
        number=1,
        position=np.array([0.0, 0.0, 0.0]),
    ),
    atom.Atom(
        symbol="O",
        number=8,
        position=np.array([0.0, 0.0, 1.0]),
    ),
]

_TEST_MOL_BASIS = molecular_basis.MolecularBasis(
    atoms=_TEST_ATOMS,
    basis_blocks=[
        basis_block.BasisBlock(
            center=np.array([0.0, 0.0, 0.0]),
            exponents=np.array([0.1], dtype=np.float64),
            cartesian_powers=np.array([[0, 0, 0]]),
            contraction_matrix=np.array([[1.0]], dtype=np.float64),
            basis_transform=np.eye(1, dtype=np.float64),
        ),
        basis_block.BasisBlock(
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
    block_slices=[slice(0, 1), slice(1, 5)],
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


def test_two_electron_matrix(mol_basis=_TEST_MOL_BASIS, P=_TEST_P):
    # Compute the two-electron Fock matrix G by brute force.
    # First compute all the two-electron integrals.
    n_basis = mol_basis.n_basis
    blocks = mol_basis.basis_blocks
    slices = mol_basis.block_slices
    V = np.zeros((n_basis, n_basis, n_basis, n_basis), dtype=np.float64)

    for i, j, k, l in itertools.product(range(len(blocks)), repeat=4):
        V_block = operators.two_electron_matrix(
            blocks[i], blocks[j], blocks[k], blocks[l], coulomb.two_electron
        )
        V[slices[i], slices[j], slices[k], slices[l]] = V_block

    # Compute G using the definition.
    # G_{ij} = sum_{kl} P_{kl} ( (ij|lk) - 0.5 (ik|lj) )
    G_expected = np.zeros((n_basis, n_basis), dtype=np.float64)
    for i, j, k, l in itertools.product(range(n_basis), repeat=4):
        G_expected[i, j] += P[k, l] * (V[i, j, l, k] - 0.5 * V[i, k, l, j])

    # Compute G using the two_electron_matrix function.
    G_result = fock.two_electron_matrix(mol_basis, P)

    np.testing.assert_allclose(G_result, G_expected, rtol=1e-7, atol=1e-7)


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

    E_result = fock.electronic_energy(H_core, F, P)

    np.testing.assert_almost_equal(E_result, E_expected, decimal=10)
