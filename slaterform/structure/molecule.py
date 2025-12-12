import dataclasses

import numpy as np

from slaterform.structure import atom


@dataclasses.dataclass
class Molecule:
    atoms: list[atom.Atom]
