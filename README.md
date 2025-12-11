![Coverage Status](./coverage.svg)

# Overview

This is a self-contained implementation of the Hartree-Fock
self-consistent field method in pure python/numpy.
This includes an implementation of the necessary electron integrals.

The only dependencies are used to load and parse data from external sources such as [basis set exchange](https://www.basissetexchange.org/)
and [pubchem](https://pubchem.ncbi.nlm.nih.gov/).

# Quick Start

Here is an example which loads water molecule geometry from pubchem and computes the electronic energy using the STO-3G basis set.


```python
import pubchempy as pcp

from chem.adapters import pubchem
from chem.adapters import bse
from chem.structure import molecule
from chem.structure import molecular_basis
from chem.hartree_fock import scf

# Load H2O geometry from pubchem.
compound = pcp.get_compounds("water", "name", record_type="3d")[0]
mol = pubchem.load_molecule(compound)

# Load the STO-3G basis set from BSE.
mol_basis = molecular_basis.build(
    mol, basis_fetcher=lambda n: bse.load("sto-3g", n))

# Callback to log progress.
def logger(s: scf.State) -> None:
    print(f'Iteration {s.iteration}: '
          f'Electronic Energy = {s.electronic_energy}, '
          f'Delta P = {s.delta_P}')

# Run SCF to solve for the electronic energy.
result = scf.solve(mol_basis, scf.Options(callback=logger))
print(f"Electronic Energy: {result.electronic_energy} H")
```

Output:

```
Iteration 0: Electronic Energy = 0.0000000000 Delta P = 5.3606682079e+00
Iteration 1: Electronic Energy = -82.3221148408 Delta P = 3.6807088051e+00
Iteration 2: Electronic Energy = -84.0275397612 Delta P = 4.1223597134e-01
Iteration 3: Electronic Energy = -84.0478104496 Delta P = 7.7251144625e-02
...
Iteration 17: Electronic Energy = -84.0488121173 Delta P = 5.4417475857e-07

Electronic Energy: -84.04881211726536 H
```


