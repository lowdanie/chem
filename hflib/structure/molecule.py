import dataclasses

import numpy as np

from hflib.structure import atom


@dataclasses.dataclass
class Molecule:
    atoms: list[atom.Atom]
