![Coverage Status](./coverage.svg)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lowdanie/hartree-fock-solver/blob/main/notebooks/geometry_optimization.ipynb)

# Slaterform

Slaterform is a self-contained Hartree-Fock engine written in pure python with numpy.
It includes a native implementation of the necessary electron integrals.

# Quick Start

Here is an example which loads water molecule geometry from
[pubchem](https://pubchem.ncbi.nlm.nih.gov/)
and estimates the electronic ground state using the STO-3G basis set from
[basis set exchange](https://www.basissetexchange.org/).


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

We can now evaluate the electron density on the points of a grid and 
save the result to a
[cube file](https://cubefile.readthedocs.io/en/latest/) so that we can render it with tools like
[3dmol](https://3dmol.org/doc/index.html).

```python
from slaterform.analysis import cube_io
from slaterform.analysis import density
from slaterform.analysis import grid as analysis_grid

grid = analysis_grid.build_bounding_grid(mol)
density_data = density.evaluate(mol_basis, result.density, grid)

with open('density.cube', 'w') as f):
    cube_io.write_cube_data(
        mol=mol, grid=grid, data=density_data,
        description="density", f=f
    )
```

Here is what the result looks like on
[3dmol](https://3dmol.org/doc/index.html):

<p align="center">
  <img src="assets/images/h2o_density_cloud.png" width="600" alt="Electron Density of Water">
</p>




