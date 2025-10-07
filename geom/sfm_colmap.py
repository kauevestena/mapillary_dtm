"""Synthetic COLMAP reconstruction scaffolding."""
from __future__ import annotations

from typing import Dict, List, Mapping

import numpy as np

from ..common_core import FrameMeta, Pose, ReconstructionResult
from .utils import heading_matrix, positions_from_frames, synthetic_ground_offsets


def run(
    seqs: Mapping[str, List[FrameMeta]],
    rng_seed: int = 4025,
) -> Dict[str, ReconstructionResult]:
    """Produce pseudo-COLMAP reconstructions, decorrelated from OpenSfM."""

    rng = np.random.default_rng(rng_seed)
    results: Dict[str, ReconstructionResult] = {}

    for seq_id, frames in seqs.items():
        if not frames:
            continue

        positions, _ = positions_from_frames(frames)
        if positions.size == 0:
            continue

        poses = {}
        points: List[np.ndarray] = []
        offsets = synthetic_ground_offsets() * np.array([1.05, 0.95, 1.0])

        for idx, frame in enumerate(frames):
            base_pos = positions[idx]
            drift = rng.normal(scale=0.07, size=3)
            pos = base_pos + drift + np.array([0.1, -0.1, 0.02])
            R = heading_matrix(positions, idx)
            yaw_jitter = rng.normal(scale=0.01)
            R = _yaw_perturb(R, yaw_jitter)
            poses[frame.image_id] = Pose(R=R, t=pos)

            for offset in offsets:
                jitter = rng.normal(scale=0.1, size=3)
                points.append(pos + offset + jitter)

        points_xyz = np.vstack(points) if points else np.zeros((0, 3), dtype=float)

        results[seq_id] = ReconstructionResult(
            seq_id=seq_id,
            frames=list(frames),
            poses=poses,
            points_xyz=points_xyz.astype(np.float32),
            source="colmap",
            metadata={"rng_seed": rng_seed, "point_count": int(points_xyz.shape[0])},
        )

    return results


def _yaw_perturb(R: np.ndarray, delta: float) -> np.ndarray:
    c, s = np.cos(delta), np.sin(delta)
    J = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return J @ R
