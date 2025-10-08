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

    pts: list[np.ndarray] = []
    wts: list[float] = []

    for alpha in alphas:
        center = (1.0 - alpha) * p0 + alpha * p1
        for offset in offsets:
            pt = center + lateral * float(offset)
            pt[2] = ground_z
            pts.append(pt.astype(np.float32))

            # Confidence is higher closer to the middle and near the lane center.
            center_weight = 1.0 - abs(alpha - 0.5) * 1.6
            lane_weight = 1.0 - abs(offset) / (max(abs(offsets).max(), 1e-3) * 1.5)
            weight = float(np.clip(0.4 + 0.6 * center_weight * lane_weight, 0.0, 1.0))
            wts.append(weight)

    if not pts:
        return SweepResult(
            points=np.zeros((0, 3), dtype=np.float32),
            weights=np.zeros((0,), dtype=np.float32),
        )

    return SweepResult(points=np.vstack(pts), weights=np.asarray(wts, dtype=np.float32))
