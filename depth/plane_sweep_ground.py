"""
Local plane-sweep restricted to plausible ground hypotheses.

This synthetic variant samples ground points along the baseline connecting two
consecutive frames. It returns candidate 3D points (in the local ENU frame
shared by the sequence) and confidence weights that later stages can use to
prioritize higher quality observations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..common_core import FrameMeta
from ..geom.utils import positions_from_frames


@dataclass(frozen=True)
class SweepResult:
    points: np.ndarray          # (N, 3) in meters, ENU-aligned
    weights: np.ndarray         # (N,) confidence weights in [0, 1]


def sweep(
    pair: Sequence[FrameMeta],
    samples_along: int = 18,
    lateral_offsets: Sequence[float] = (-2.0, -0.7, 0.7, 2.0),
    assumed_cam_height: float = 1.6,
) -> SweepResult:
    """Generate synthetic ground points from a frame pair.

    Parameters
    ----------
    pair:
        Two frames (typically consecutive) that define the sweep baseline.
    samples_along:
        Number of samples between the two camera centers.
    lateral_offsets:
        Meters offset from the baseline to emulate multi-lane coverage.
    assumed_cam_height:
        Approximate camera height above the ground plane.
    """

    frames = list(pair)
    if len(frames) != 2:
        raise ValueError("plane-sweep expects exactly two frames")

    positions, _ = positions_from_frames(frames)
    if positions.shape[0] < 2:
        return SweepResult(
            points=np.zeros((0, 3), dtype=np.float32),
            weights=np.zeros((0,), dtype=np.float32),
        )

    p0, p1 = positions[0], positions[1]
    baseline = p1 - p0
    baseline_norm = np.linalg.norm(baseline[:2])
    if baseline_norm < 1e-6:
        lateral = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        lateral = np.array([-baseline[1], baseline[0], 0.0], dtype=float)
        lateral /= np.linalg.norm(lateral)

    ground_z = float(np.mean([p0[2], p1[2]]) - assumed_cam_height)
    alphas = np.linspace(0.0, 1.0, samples_along, dtype=np.float32)
    offsets = np.asarray(list(lateral_offsets), dtype=np.float32)

    # Vectorized point generation using meshgrid
    alpha_grid, offset_grid = np.meshgrid(alphas, offsets, indexing='ij')
    
    # Centers shape: (samples_along, 1, 3)
    # Lateral shape: (1, offsets, 3)
    alpha_expanded = alpha_grid[..., np.newaxis]
    centers = (1.0 - alpha_expanded) * p0 + alpha_expanded * p1
    
    offset_expanded = offset_grid[..., np.newaxis]
    lateral_shift = lateral * offset_expanded
    
    pts_grid = centers + lateral_shift
    pts_grid[..., 2] = ground_z
    
    # Vectorized weights
    center_weight = 1.0 - np.abs(alpha_grid - 0.5) * 1.6
    lane_weight = 1.0 - np.abs(offset_grid) / (max(np.abs(offsets).max(), 1e-3) * 1.5)
    weights_grid = np.clip(0.4 + 0.6 * center_weight * lane_weight, 0.0, 1.0)
    
    pts = pts_grid.reshape(-1, 3).astype(np.float32)
    wts = weights_grid.reshape(-1).astype(np.float32)

    if pts.size == 0:
        return SweepResult(
            points=np.zeros((0, 3), dtype=np.float32),
            weights=np.zeros((0,), dtype=np.float32),
        )

    return SweepResult(points=pts, weights=wts)
