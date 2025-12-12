import dataclasses

import numpy as np


@dataclasses.dataclass
class Atom:
    symbol: str
    number: int  # Atomic number

    # Position in Bohr units
    # shape (3,)
    position: np.ndarray
