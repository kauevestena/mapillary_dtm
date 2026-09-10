"""Rasterize local measurements only where a full cell has observed support."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .estimation import FitPolicy, Plane, angular_bound, fit_plane


@dataclass(frozen=True)
class Grid:
    west: float
    north: float
    resolution: float
    width: int
    height: int

    @classmethod
    def from_points(cls, points: np.ndarray, resolution: float, max_cells: int):
        if not np.isfinite(resolution) or resolution <= 0:
            raise ValueError("Grid resolution must be finite and positive")
        if not isinstance(max_cells, int) or max_cells < 1:
            raise ValueError("max_cells must be a positive integer")
        if len(points) == 0:
            raise ValueError("No observed surface points available for gridding")
        west, south = np.floor(np.min(points[:, :2], axis=0) / resolution) * resolution
        east, north = np.ceil(np.max(points[:, :2], axis=0) / resolution) * resolution
        width = max(1, int(round((east-west)/resolution)))
        height = max(1, int(round((north-south)/resolution)))
        if width * height > max_cells:
            raise ValueError(f"Grid requires {width*height:,} cells; limit is {max_cells:,}. Tile the project.")
        return cls(float(west), float(north), resolution, width, height)

    def center(self, row: int, col: int) -> np.ndarray:
        return np.array([self.west + (col + 0.5) * self.resolution,
                         self.north - (row + 0.5) * self.resolution])

    def index(self, xy: np.ndarray) -> tuple[int, int] | None:
        col = int(np.floor((xy[0] - self.west) / self.resolution))
        row = int(np.floor((self.north - xy[1]) / self.resolution))
        return (row, col) if 0 <= row < self.height and 0 <= col < self.width else None


@dataclass
class Measurement:
    plane: Plane
    source_id: str
    evidence_group: str  # repeated reconstructions of the same images share this
    bound_deg: float | None
    registration_rmse_m: float
    point_ids: list[str]


@dataclass
class Cell:
    gradient: np.ndarray
    bound_deg: float | None
    fit_precision_deg: float
    independent_groups: int
    sources: list[str]
    observations: list[Measurement]


def estimate_cells(points: np.ndarray, ids: list[str], grid: Grid, policy: FitPolicy,
                   *, source_id: str, evidence_group: str, vertical_uncertainty: float | None,
                   geometry_uncertainty: float | None, registration_rmse_m: float):
    tree = cKDTree(points[:, :2])
    # A candidate center must be close to an observed point. Avoid scanning an
    # entire bounding rectangle (e.g. the unobserved interior of a road loop).
    candidate_keys = set()
    radius_cells = int(np.ceil(policy.max_gap_m / grid.resolution)) + 1
    for point in points:
        col = int(np.floor((point[0] - grid.west) / grid.resolution))
        row = int(np.floor((grid.north - point[1]) / grid.resolution))
        for r in range(max(0, row-radius_cells), min(grid.height, row+radius_cells+1)):
            for c in range(max(0, col-radius_cells), min(grid.width, col+radius_cells+1)):
                candidate_keys.add((r, c))
    cells, reasons = {}, Counter()
    for key in sorted(candidate_keys):
        center = grid.center(*key)
        indices = tree.query_ball_point(center, policy.radius_m)
        plane, reason = fit_plane(points[indices], center, policy, cell_size_m=grid.resolution)
        reasons[reason] += 1
        if plane is None:
            continue
        cells[key] = Measurement(plane, source_id, evidence_group,
                                 angular_bound(plane, vertical_uncertainty, geometry_uncertainty),
                                 registration_rmse_m, [ids[i] for i in indices])
    return cells, dict(reasons)


def fuse_cells(candidates: dict[tuple[int, int], list[Measurement]], max_disagreement_deg: float):
    """Compare full normals, not slope magnitudes. Conflicts become unknown.

    Shared-image backends are not independent evidence. Averaging never shrinks
    the uncertainty allowance below the largest input allowance.
    """
    if not np.isfinite(max_disagreement_deg) or not 0 < max_disagreement_deg < 90:
        raise ValueError("max_disagreement_deg must be in (0, 90)")
    accepted, conflicts = {}, 0
    for key, observations in candidates.items():
        normals = np.asarray([obs.plane.normal for obs in observations])
        disagreement = float(np.degrees(np.arccos(np.clip(normals @ normals.T, -1, 1).min())))
        if disagreement > max_disagreement_deg:
            conflicts += 1
            continue
        # One vote per capture/evidence group; adding another SfM implementation
        # on the same sequence cannot outvote an independent sequence.
        groups = {}
        for obs in observations:
            groups.setdefault(obs.evidence_group, []).append(obs.plane.gradient)
        gradient = np.mean([np.mean(values, axis=0) for values in groups.values()], axis=0)
        bounds = [obs.bound_deg for obs in observations]
        bound = None if any(value is None for value in bounds) else max(bounds) + disagreement
        accepted[key] = Cell(gradient, bound,
                             max(obs.plane.fit_precision_deg for obs in observations),
                             len(groups), sorted({obs.source_id for obs in observations}), observations)
    return accepted, conflicts
