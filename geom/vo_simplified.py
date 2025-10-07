"""Simplified VO chain that keeps relative motion up-to-scale."""
from __future__ import annotations

from typing import Dict, List, Mapping

import numpy as np

from ..common_core import FrameMeta, Pose, ReconstructionResult
from .utils import heading_matrix, positions_from_frames


def run(
    seqs: Mapping[str, List[FrameMeta]],
    rng_seed: int = 3025,
) -> Dict[str, ReconstructionResult]:
    """Return relative trajectories (no absolute scale)."""

    rng = np.random.default_rng(rng_seed)
    results: Dict[str, ReconstructionResult] = {}

    for seq_id, frames in seqs.items():
        if not frames:
            continue

        positions, _ = positions_from_frames(frames)
        if positions.size == 0:
            continue

        # Normalize to start at origin and unit-average step norm.
        positions -= positions[0]
        step_norms = np.linalg.norm(np.diff(positions, axis=0), axis=1) if positions.shape[0] > 1 else np.array([1.0])
        scale = float(np.mean(step_norms)) if step_norms.size else 1.0
        if scale <= 1e-6:
            scale = 1.0
        rel_positions = positions / scale

        poses = {}
        for idx, frame in enumerate(frames):
            pos = rel_positions[idx] + rng.normal(scale=0.01, size=3)
            R = heading_matrix(rel_positions, idx)
            poses[frame.image_id] = Pose(R=R, t=pos)

        results[seq_id] = ReconstructionResult(
            seq_id=seq_id,
            frames=list(frames),
            poses=poses,
            points_xyz=np.zeros((0, 3), dtype=np.float32),
            source="vo",
            metadata={"scale": scale, "rng_seed": rng_seed},
        )

    return results
