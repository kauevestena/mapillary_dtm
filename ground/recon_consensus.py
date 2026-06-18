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

    import pandas as pd

    data = []
    def _accumulate(points: Sequence[GroundPoint] | None, label: str) -> None:
        if not points:
            return
        for gp in points:
            if gp is None:
                continue
            data.append({
                "ix": int(np.floor(gp.x / grid)),
                "iy": int(np.floor(gp.y / grid)),
                "label": label,
                "x": gp.x, "y": gp.y, "z": gp.z,
                "uncertainty_m": gp.uncertainty_m,
                "sem_prob": gp.sem_prob,
                "tri_angle_deg": gp.tri_angle_deg
            })

    _accumulate(ptsA, "A")
    _accumulate(ptsB, "B")
    _accumulate(ptsC, "C")

    # Determine minimum agreeing sources. Always require 2 for consensus.
    min_sources = 2

    if not data:
        return []

    df = pd.DataFrame(data)
    
    # Vectorized filtering: keep grid cells with at least min_sources distinct sources
    grouped = df.groupby(["ix", "iy"])
    filtered_df = df[grouped["label"].transform("nunique") >= min_sources]

    consensus: List[ConsensusPoint] = []
    if filtered_df.empty:
        return consensus

    for (ix, iy), group in filtered_df.groupby(["ix", "iy"]):
        by_source = group.groupby("label")["z"].mean().to_dict()
        
        agreeing_sources = _sources_within_threshold(by_source, dz_thr)
        if len(agreeing_sources) < min_sources:
            continue

        supporting = group[group["label"].isin(agreeing_sources)]
        if supporting.empty:
            continue

        weights = np.clip(1.0 / np.maximum(supporting["uncertainty_m"], 1e-3), 0.1, 20.0)
        z_pct = np.percentile(supporting["z"], constants.LOWER_ENVELOPE_Q * 100.0)
        
        centroid_x = np.average(supporting["x"], weights=weights)
        centroid_y = np.average(supporting["y"], weights=weights)
        avg_sem = np.average(supporting["sem_prob"], weights=weights)
        avg_unc = np.average(supporting["uncertainty_m"], weights=weights)
        
        tri_angles = supporting["tri_angle_deg"].dropna()
        tri_angle = tri_angles.mean() if not tri_angles.empty else None

        consensus.append({
            "x": float(centroid_x),
            "y": float(centroid_y),
            "z": float(z_pct),
            "sources": sorted(agreeing_sources),
            "support": len(supporting),
            "sem_prob": float(np.clip(avg_sem, 0.0, 1.0)),
            "uncertainty": float(np.clip(avg_unc, 0.05, 0.6)),
            "tri_angle_deg": float(tri_angle) if pd.notna(tri_angle) else None,
        })

    return consensus


def _sources_within_threshold(
    source_heights: Mapping[str, float],
    dz_thr: float,
) -> List[str]:
    labels = list(source_heights.keys())
    if len(labels) == 1:
        return labels
    agreeing: set[str] = set()
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            li, lj = labels[i], labels[j]
            if abs(source_heights[li] - source_heights[lj]) <= dz_thr:
                agreeing.add(li)
                agreeing.add(lj)
    return sorted(agreeing)
