"""
Robust lower-envelope 2.5D fusion to a 0.5 m grid; confidence estimation.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .. import constants

ConsensusPoint = Mapping[str, object]


def fuse(
    points: Sequence[ConsensusPoint] | None,
    grid_res: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse consensus ground points into a gridded DTM + confidence map."""

    if not points:
        return (
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
        )

    res = float(grid_res or constants.GRID_RES_M)

    xs = np.asarray([float(pt["x"]) for pt in points], dtype=np.float64)
    ys = np.asarray([float(pt["y"]) for pt in points], dtype=np.float64)
    if xs.size == 0 or ys.size == 0:
        return (
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
        )

    ix_min = int(np.floor(xs.min() / res))
    ix_max = int(np.floor(xs.max() / res))
    iy_min = int(np.floor(ys.min() / res))
    iy_max = int(np.floor(ys.max() / res))

    width = max(1, ix_max - ix_min + 1)
    height = max(1, iy_max - iy_min + 1)

    height_lists: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    weight_lists: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    sem_lists: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    src_counts: Dict[Tuple[int, int], List[int]] = defaultdict(list)

    for pt in points:
        x = float(pt["x"])
        y = float(pt["y"])
        z = float(pt["z"])
        sources = pt.get("sources") or []
        sem_prob = float(pt.get("sem_prob", 0.7))
        uncertainty = float(pt.get("uncertainty", 0.25))

        ix = int(np.floor(x / res)) - ix_min
        iy = int(np.floor(y / res)) - iy_min
        if ix < 0 or iy < 0 or ix >= width or iy >= height:
            continue

        key = (iy, ix)
        height_lists[key].append(z)
        weight_lists[key].append(_weight_from_uncertainty(uncertainty))
        sem_lists[key].append(sem_prob)
        src_counts[key].append(len(sources))

    dtm = np.full((height, width), np.nan, dtype=np.float32)
    confidence = np.zeros((height, width), dtype=np.float32)

    for (iy, ix), heights in height_lists.items():
        arr = np.asarray(heights, dtype=np.float32)
        if arr.size == 0:
            continue

        weights = np.asarray(weight_lists[(iy, ix)], dtype=np.float32)
        sems = np.asarray(sem_lists[(iy, ix)], dtype=np.float32)
        srcs = np.asarray(src_counts[(iy, ix)], dtype=np.float32)

        dtm[iy, ix] = float(np.percentile(arr, constants.LOWER_ENVELOPE_Q * 100.0))

        avg_sem = float(np.clip(np.average(sems, weights=weights), 0.0, 1.0))
        avg_weight = float(np.mean(weights))
        sample_count = arr.size
        avg_sources = float(np.mean(srcs))

        conf = (
            0.35
            + 0.35 * min(sample_count / 4.0, 1.0)
            + 0.20 * min(avg_weight / 4.0, 1.0)
            + 0.10 * min(avg_sources / 3.0, 1.0)
        )
        conf *= 0.6 + 0.4 * avg_sem
        confidence[iy, ix] = float(np.clip(conf, 0.0, 1.0))

    dtm = np.nan_to_num(dtm, nan=np.nanmean(dtm[np.isfinite(dtm)]) if np.isfinite(dtm).any() else 0.0)
    return dtm.astype(np.float32), confidence.astype(np.float32)


def _weight_from_uncertainty(uncertainty: float) -> float:
    uncertainty = max(float(uncertainty), 1e-3)
    return float(np.clip(1.0 / uncertainty, 0.5, 10.0))
