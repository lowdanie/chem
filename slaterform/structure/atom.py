import dataclasses

import numpy as np
import numpy.typing as npt


@dataclasses.dataclass
class Atom:
    symbol: str
    number: int  # Atomic number

    # Position in Bohr units
    # shape (3,)
    position: npt.NDArray[np.float64]
