![Coverage Status](./coverage.svg)

# Slaterform

Slaterform is a self-contained Hartree-Fock engine written in pure python with numpy.
It includes a native implementation of the necessary electron integrals.

# Quick Start

Here is an example which loads water molecule geometry from pubchem and computes the electronic energy using the STO-3G basis set.


```python
import pubchempy as pcp

from slaterform.adapters import pubchem
from slaterform.adapters import bse
from slaterform.structure import molecule
from slaterform.structure import molecular_basis
from slaterform.hartree_fock import scf

# Load H2O geometry from pubchem.
compound = pcp.get_compounds("water", "name", record_type="3d")[0]
mol = pubchem.load_molecule(compound)

# Load the STO-3G basis set from BSE.
mol_basis = molecular_basis.build(
    mol, basis_fetcher=lambda n: bse.load("sto-3g", n))

# Callback to log progress.
def logger(s: scf.State) -> None:
    print(f'Iteration {s.iteration}: '
          f'Total Energy = {s.total_energy}, '
          f'Delta P = {s.delta_P}')

# Run SCF to solve for the energy.
result = scf.solve(mol_basis, scf.Options(callback=logger))

print(f"Total Energy: {result.total_energy} H")
print(f"Electronic Energy: {result.electronic_energy} H")
```

Output:

```
Iteration 1: Total Energy = -73.23775033094817, Delta P = 5.360668207876737
Iteration 2: Total Energy = -74.94317525135683, Delta P = 3.6807088050698793
Iteration 3: Total Energy = -74.96344593966914, Delta P = 0.41223597134446294
...
Iteration 17: Total Energy = -74.96444760738018, Delta P = 1.245141361212375e-06
Iteration 18: Total Energy = -74.96444760738025, Delta P = 5.441747585731223e-07

Total Energy: -74.96444760738025 H
Electronic Energy: -84.04881211726543 H
```


