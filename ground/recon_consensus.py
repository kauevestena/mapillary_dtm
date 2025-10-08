"""
Cross-validate ground reconstructions (A/B/C) and keep consensus points.

Each input list is expected to contain :class:`GroundPoint` objects created by
``ground_extract_3d.label_and_filter_points``. The consensus stage voxelizes
the scene at the pipeline grid resolution, checks that at least two distinct
sources agree in height within ``DZ_MAX_M``, and emits a blended point record
with averaged metadata for downstream fusion.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .. import constants
from ..common_core import GroundPoint

ConsensusPoint = Dict[str, object]


def agree(
    ptsA: Sequence[GroundPoint] | None,
    ptsB: Sequence[GroundPoint] | None,
    ptsC: Sequence[GroundPoint] | None,
    grid_res: float | None = None,
    dz_max: float | None = None,
) -> List[ConsensusPoint]:
    """Return consensus ground points satisfying multi-source agreement."""

    grid = float(grid_res or constants.GRID_RES_M)
    dz_thr = float(dz_max or constants.DZ_MAX_M)

    buckets: Dict[Tuple[int, int], List[Tuple[str, GroundPoint]]] = defaultdict(list)

    def _accumulate(points: Sequence[GroundPoint] | None, label: str) -> None:
        if not points:
            return
        for gp in points:
            if gp is None:
                continue
            ix = int(np.floor(gp.x / grid))
            iy = int(np.floor(gp.y / grid))
            buckets[(ix, iy)].append((label, gp))

    _accumulate(ptsA, "A")
    _accumulate(ptsB, "B")
    _accumulate(ptsC, "C")

    consensus: List[ConsensusPoint] = []
    for (ix, iy), records in buckets.items():
        if len(records) < 2:
            continue

        by_source: Dict[str, List[GroundPoint]] = defaultdict(list)
        for label, gp in records:
            by_source[label].append(gp)

        if len(by_source) < 2:
            continue

        source_heights = {
            label: np.mean([gp.z for gp in items]) for label, items in by_source.items()
        }

        agreeing_sources = _sources_within_threshold(source_heights, dz_thr)
        if len(agreeing_sources) < 2:
            continue

        supporting_points = [
            gp
            for label, gps in by_source.items()
            if label in agreeing_sources
            for gp in gps
        ]
        if not supporting_points:
            continue

        xyz = np.array([[gp.x, gp.y, gp.z] for gp in supporting_points], dtype=np.float64)
        weights = np.clip(
            np.array([1.0 / max(gp.uncertainty_m, 1e-3) for gp in supporting_points], dtype=np.float64),
            0.1,
            20.0,
        )

        z_percentile = float(np.percentile(xyz[:, 2], constants.LOWER_ENVELOPE_Q * 100.0))
        centroid = np.average(xyz[:, :2], axis=0, weights=weights)
        avg_sem = float(np.average([gp.sem_prob for gp in supporting_points], weights=weights))
        avg_unc = float(np.average([gp.uncertainty_m for gp in supporting_points], weights=weights))
        tri_values = [gp.tri_angle_deg for gp in supporting_points if gp.tri_angle_deg is not None]
        tri_angle = float(np.mean(tri_values)) if tri_values else None

        consensus.append(
            {
                "x": float(centroid[0]),
                "y": float(centroid[1]),
                "z": z_percentile,
                "sources": sorted(agreeing_sources),
                "support": len(supporting_points),
                "sem_prob": float(np.clip(avg_sem, 0.0, 1.0)),
                "uncertainty": float(np.clip(avg_unc, 0.05, 0.6)),
                "tri_angle_deg": tri_angle,
            }
        )

    return consensus


def _sources_within_threshold(
    source_heights: Mapping[str, float],
    dz_thr: float,
) -> List[str]:
    labels = list(source_heights.keys())
    agreeing: set[str] = set()
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            li, lj = labels[i], labels[j]
            if abs(source_heights[li] - source_heights[lj]) <= dz_thr:
                agreeing.add(li)
                agreeing.add(lj)
    return sorted(agreeing)
