import dataclasses
import pytest

import numpy as np
import numpy.typing as npt

from hflib.analysis import grid as analysis_grid
from hflib.structure import atom
from hflib.structure import molecule


@dataclasses.dataclass
class _GeneratePointsTestCase:
    grid: analysis_grid.RegularGrid
    expected_points: npt.NDArray[np.float64]  # shape: grid.dims + (, 3)


@dataclasses.dataclass
class _BuildBoundingGridTestCase:
    mol: molecule.Molecule
    spacing: float
    padding: float
    expected_grid: analysis_grid.RegularGrid


@pytest.mark.parametrize(
    "case",
    [
        _GeneratePointsTestCase(
            grid=analysis_grid.RegularGrid(
                origin=np.array([0.0, 0.0, 0.0]),
                spacing=1.0,
                dims=(2, 2, 3),
            ),
            expected_points=np.array(
                [
                    [
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
                        [[0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 2.0]],
                    ],
                    [
                        [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, 2.0]],
                        [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 2.0]],
                    ],
                ]
            ),
        ),
    ],
)
def test_generate_points(case: _GeneratePointsTestCase) -> None:
    points = analysis_grid.generate_points(case.grid)
    assert points.shape == case.expected_points.shape
    np.testing.assert_allclose(points, case.expected_points)


@pytest.mark.parametrize(
    "case",
    [
        _BuildBoundingGridTestCase(
            mol=molecule.Molecule(
                atoms=[
                    atom.Atom(
                        symbol="H",
                        number=1,
                        position=np.array([0.0, -1.0, 0.0]),
                    ),
                    atom.Atom(
                        symbol="O",
                        number=8,
                        position=np.array([1.0, 2.0, 0.0]),
                    ),
                ],
            ),
            spacing=0.5,
            padding=2.0,
            # nx = 3 - (-2) / 0.5 = 10
            # ny = 4 - (-3) / 0.5 = 14
            # nz = 2 - (-2) / 0.5 = 8
            expected_grid=analysis_grid.RegularGrid(
                origin=np.array([-2.0, -3.0, -2.0]),
                spacing=0.5,
                dims=(10, 14, 8),
            ),
        ),
    ],
)
def test_build_bounding_grid(case: _BuildBoundingGridTestCase) -> None:
    grid = analysis_grid.build_bounding_grid(
        case.mol, case.padding, case.spacing
    )
    np.testing.assert_allclose(grid.origin, case.expected_grid.origin)
    assert grid.spacing == case.expected_grid.spacing
    assert grid.dims == case.expected_grid.dims
