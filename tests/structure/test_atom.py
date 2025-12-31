import numpy as np

from tests.jax_utils import pytree_utils

import slaterform as sf


def test_atom_pytree():
    atom = sf.Atom(
        symbol="H",
        number=1,
        position=np.array([0.0, 0.0, 0.0]),
        shells=[
            sf.ContractedGTO(
                primitive_type=sf.PrimitiveType.CARTESIAN,
                angular_momentum=(0,),
                exponents=np.array([1.0], dtype=np.float64),
                coefficients=np.array([[2.0]], dtype=np.float64),
            )
        ],
    )
    pytree_utils.assert_valid_pytree(atom)
