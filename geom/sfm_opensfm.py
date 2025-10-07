"""Synthetic OpenSfM reconstruction scaffolding."""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping

import numpy as np

from ..common_core import FrameMeta, Pose, ReconstructionResult
from .utils import heading_matrix, positions_from_frames, synthetic_ground_offsets


def run(
    seqs: Mapping[str, List[FrameMeta]],
    rng_seed: int = 2025,
) -> Dict[str, ReconstructionResult]:
    """Produce lightweight pseudo-reconstructions for testing/integration."""

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
        offsets = synthetic_ground_offsets()

        for idx, frame in enumerate(frames):
            base_pos = positions[idx]
            noise = rng.normal(scale=0.05, size=3)
            pos = base_pos + noise
            R = heading_matrix(positions, idx)
            poses[frame.image_id] = Pose(R=R, t=pos)

            for offset in offsets:
                jitter = rng.normal(scale=0.08, size=3)
                points.append(pos + offset + jitter)

        points_xyz = np.vstack(points) if points else np.zeros((0, 3), dtype=float)

        results[seq_id] = ReconstructionResult(
            seq_id=seq_id,
            frames=list(frames),
            poses=poses,
            points_xyz=points_xyz.astype(np.float32),
            source="opensfm",
            metadata={"rng_seed": rng_seed, "point_count": int(points_xyz.shape[0])},
        )

    return results
