import pytest

import numpy as np

from hartree_fock import density


def test_closed_shell_matrix():
    num_electrons = 4
    C = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        dtype=np.float64,
    )

    expected_P = 2 * np.array(
        [
            [5, 14, 23],
            [14, 41, 68],
            [23, 68, 113],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        density.closed_shell_matrix(C, num_electrons),
        expected_P,
        rtol=1e-7,
        atol=1e-7,
    )


def test_closed_shell_matrix_odd_electrons():
    num_electrons = 3
    C = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        density.closed_shell_matrix(C, num_electrons)
