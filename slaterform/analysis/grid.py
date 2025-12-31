import dataclasses
import numpy as np
import numpy.typing as npt


from slaterform.structure import molecule


@dataclasses.dataclass
class RegularGrid:
    """Represents a regularly spaced 3D grid for volumetric data."""

    origin: np.ndarray  # Shape (3,)
    dims: tuple[int, int, int]  # (nx, ny, nz)
    spacing: float  # Bohr (assuming cubic voxels)


def generate_points(grid: RegularGrid) -> np.ndarray:
    """Generate the 3D points of the regular grid.

    Args:
        grid: The RegularGrid defining the grid parameters.

    Returns:
        An array points of shape (nx, ny, nz, 3) satisfying:
        points[i, j, k] = [x_i, y_j, z_k]
    """
    nx, ny, nz = grid.dims

    x = np.arange(nx) * grid.spacing + grid.origin[0]
    y = np.arange(ny) * grid.spacing + grid.origin[1]
    z = np.arange(nz) * grid.spacing + grid.origin[2]

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    return np.stack((X, Y, Z), axis=-1)


def build_bounding_grid(
    mol: molecule.Molecule, padding: float = 3.0, spacing: float = 0.2
) -> RegularGrid:
    """Build a bounding regular grid around the given molecule.

    Args:
        mol: The molecule to bound.
        padding: The extra space to add around the molecule in Bohr.
        spacing: The desired spacing between grid points in Bohr.

    Returns:
        A RegularGrid that bounds the input points.
    """
    coords = np.array([a.position for a in mol.atoms])
    min_coords = np.min(coords, axis=0) - padding
    max_coords = np.max(coords, axis=0) + padding

    dims = np.ceil((max_coords - min_coords) / spacing).astype(int)

    return RegularGrid(origin=min_coords, dims=tuple(dims), spacing=spacing)
