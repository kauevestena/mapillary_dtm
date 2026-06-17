"""
Robust lower-envelope 2.5D fusion to a 0.5 m grid; confidence estimation.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .. import constants

ConsensusPoint = Mapping[str, object]


@dataclass(frozen=True)
class GridSpec:
    ix_min: int
    iy_min: int
    width: int
    height: int
    res: float


def fuse(
    points: Sequence[ConsensusPoint] | None,
    grid_res: float | None = None,
    *,
    return_grid: bool = False,
    grid: "GridSpec | None" = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, GridSpec]:
    """Fuse consensus ground points into a gridded DTM + confidence map.

    Parameters
    ----------
    points:
        Consensus ground points (dicts with ``"x"``, ``"y"``, ``"z"`` keys).
    grid_res:
        Grid cell size in metres.  Ignored when *grid* is supplied.
    return_grid:
        If ``True`` the function returns a 3-tuple ``(dtm, confidence, grid)``.
    grid:
        Pre-computed :class:`GridSpec` to use instead of deriving one from
        the data extent.  When supplied the output arrays have exactly the
        dimensions ``(grid.height, grid.width)`` and points that fall outside
        the grid are silently discarded.  Passing this allows the caller to
        pin the grid to the *consensus* point extent before TIN-augmented
        samples inflate the bounding box."""

    if not points:
        grid = grid or GridSpec(0, 0, 1, 1, float(grid_res or constants.GRID_RES_M))
        if return_grid:
            return (
                np.full((grid.height, grid.width), np.nan, dtype=np.float32),
                np.zeros((grid.height, grid.width), dtype=np.float32),
                grid,
            )
        return (
            np.full((grid.height, grid.width), np.nan, dtype=np.float32),
            np.zeros((grid.height, grid.width), dtype=np.float32),
        )

    res = float(grid.res if grid is not None else (grid_res or constants.GRID_RES_M))

    xs = np.asarray([float(pt["x"]) for pt in points], dtype=np.float64)
    ys = np.asarray([float(pt["y"]) for pt in points], dtype=np.float64)
    if xs.size == 0 or ys.size == 0:
        grid = grid or GridSpec(0, 0, 1, 1, res)
        if return_grid:
            return (
                np.full((grid.height, grid.width), np.nan, dtype=np.float32),
                np.zeros((grid.height, grid.width), dtype=np.float32),
                grid,
            )
        return (
            np.full((grid.height, grid.width), np.nan, dtype=np.float32),
            np.zeros((grid.height, grid.width), dtype=np.float32),
        )

    # Use the pre-supplied grid when available; otherwise derive from data extent.
    # Passing a pre-computed grid prevents TIN-augmented samples from inflating
    # the bounding box relative to the consensus-point coverage.
    if grid is None:
        ix_min = int(np.floor(xs.min() / res))
        ix_max = int(np.floor(xs.max() / res))
        iy_min = int(np.floor(ys.min() / res))
        iy_max = int(np.floor(ys.max() / res))
        width = max(1, ix_max - ix_min + 1)
        height = max(1, iy_max - iy_min + 1)
        grid = GridSpec(ix_min=ix_min, iy_min=iy_min, width=width, height=height, res=res)

    ix_min = grid.ix_min
    iy_min = grid.iy_min
    width = grid.width
    height = grid.height

    import pandas as pd

    # Vectorize point extraction and grid mapping
    df = pd.DataFrame([{
        "x": float(pt["x"]), "y": float(pt["y"]), "z": float(pt["z"]),
        "sem_prob": float(pt.get("sem_prob", 0.7)),
        "uncertainty": float(pt.get("uncertainty", 0.25)),
        "src_count": len(pt.get("sources") or [])
    } for pt in points])

    df["weight"] = np.clip(1.0 / np.maximum(df["uncertainty"], 1e-3), 0.5, 10.0)
    df["ix"] = (np.floor(df["x"] / res) - ix_min).astype(int)
    df["iy"] = (np.floor(df["y"] / res) - iy_min).astype(int)

    mask = (df["ix"] >= 0) & (df["iy"] >= 0) & (df["ix"] < width) & (df["iy"] < height)
    df = df[mask]

    dtm = np.full((height, width), np.nan, dtype=np.float32)
    confidence = np.zeros((height, width), dtype=np.float32)

    if not df.empty:
        grouped = df.groupby(["iy", "ix"])
        
        # DTM: lower envelope quantile of heights
        z_q = grouped["z"].quantile(constants.LOWER_ENVELOPE_Q)
        
        # Confidence components
        counts = grouped.size()
        avg_weights = grouped["weight"].mean()
        avg_srcs = grouped["src_count"].mean()
        
        # Weighted average of semantics
        wt_sem = (df["weight"] * df["sem_prob"]).groupby([df["iy"], df["ix"]]).sum()
        wt_sum = grouped["weight"].sum()
        avg_sems = (wt_sem / wt_sum).clip(0.0, 1.0)
        
        conf_term = (
            0.35 
            + 0.35 * np.minimum(counts / 4.0, 1.0)
            + 0.20 * np.minimum(avg_weights / 4.0, 1.0)
            + 0.10 * np.minimum(avg_srcs / 3.0, 1.0)
        )
        conf = (conf_term * (0.6 + 0.4 * avg_sems)).clip(0.0, 1.0)
        
        iys = z_q.index.get_level_values("iy").values
        ixs = z_q.index.get_level_values("ix").values
        
        dtm[iys, ixs] = z_q.values.astype(np.float32)
        confidence[iys, ixs] = conf.values.astype(np.float32)

    if return_grid:
        return dtm.astype(np.float32), confidence.astype(np.float32), grid
    return dtm.astype(np.float32), confidence.astype(np.float32)


def _weight_from_uncertainty(uncertainty: float) -> float:
    uncertainty = max(float(uncertainty), 1e-3)
    return float(np.clip(1.0 / uncertainty, 0.5, 10.0))


def _grid_from_points(
    points: Sequence[ConsensusPoint] | None,
    grid_res: float | None = None,
) -> GridSpec:
    """Derive a :class:`GridSpec` from a sequence of consensus points.

    This is a convenience helper for callers that need to compute the grid
    *before* passing additional (e.g. TIN-augmented) points to :func:`fuse`,
    so that the output raster dimensions are not inflated by extrapolated
    samples that lie outside the actual data coverage.

    Returns a 1×1 fallback grid when *points* is empty or ``None``.
    """
    res = float(grid_res or constants.GRID_RES_M)
    if not points:
        return GridSpec(ix_min=0, iy_min=0, width=1, height=1, res=res)

    xs = np.asarray([float(pt["x"]) for pt in points], dtype=np.float64)
    ys = np.asarray([float(pt["y"]) for pt in points], dtype=np.float64)
    if xs.size == 0 or ys.size == 0:
        return GridSpec(ix_min=0, iy_min=0, width=1, height=1, res=res)

    ix_min = int(np.floor(xs.min() / res))
    ix_max = int(np.floor(xs.max() / res))
    iy_min = int(np.floor(ys.min() / res))
    iy_max = int(np.floor(ys.max() / res))
    width = max(1, ix_max - ix_min + 1)
    height = max(1, iy_max - iy_min + 1)
    return GridSpec(ix_min=ix_min, iy_min=iy_min, width=width, height=height, res=res)
