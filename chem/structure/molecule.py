import dataclasses

import numpy as np

from chem.structure import atom


@dataclasses.dataclass
class Molecule:
    atoms: list[atom.Atom]
