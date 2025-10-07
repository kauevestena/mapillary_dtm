"""Curb / edge / lane extraction heuristics for slope-preserving breaklines."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np

from ..common_core import FrameMeta

log = logging.getLogger(__name__)


@dataclass
class CurbLine:
    seq_id: str
    image_id: str
    xy_norm: List[tuple[float, float]]  # (x,y) normalized [0,1]
    confidence: float


def extract_curbs_and_lanes(
    seqs: Mapping[str, List[FrameMeta]],
    mask_dir: Path | str = Path("cache/masks"),
    prob_band: tuple[float, float] = (0.45, 0.6),
    min_support: float = 0.3,
) -> Dict[str, List[CurbLine]]:
    """Derive simple curb/edge polylines from ground mask gradients.

    The heuristic searches for horizontal transitions in the ground
    probability map around ``prob_band``. For each image column where a
    candidate is found, we record the normalized coordinate. Columns below
    ``min_support`` coverage are discarded.
    """

    mask_path = Path(mask_dir)
    curbs: Dict[str, List[CurbLine]] = {}

    for seq_id, frames in seqs.items():
        lines: List[CurbLine] = []
        for frame in frames:
            mask_file = mask_path / f"{frame.image_id}.npz"
            arr = _load_mask(mask_file)
            if arr is None:
                continue

            band = _extract_band(arr, prob_band)
            if band is None:
                continue

            xs, ys = band
            if len(xs) / arr.shape[1] < min_support:
                continue

            poly = list(zip(xs, ys))
            lines.append(CurbLine(seq_id=seq_id, image_id=frame.image_id, xy_norm=poly,
                                  confidence=min(1.0, len(xs) / arr.shape[1])))

        if lines:
            curbs[seq_id] = lines

    return curbs


def _load_mask(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        with np.load(path) as data:
            prob = data.get("prob")
            if prob is None:
                return None
            arr = np.asarray(prob, dtype=np.float32)
            if arr.ndim != 2:
                return None
            return np.clip(arr, 0.0, 1.0)
    except Exception as exc:  # pragma: no cover
        log.warning("Failed to read mask %s: %s", path, exc)
        return None


def _extract_band(arr: np.ndarray, band: tuple[float, float]) -> Optional[tuple[List[float], List[float]]]:
    low, high = band
    if low >= high:
        raise ValueError("Invalid prob_band ordering")
    mask = (arr >= low) & (arr <= high)
    if not mask.any():
        return None

    rows, cols = arr.shape
    xs: List[float] = []
    ys: List[float] = []
    for c in range(cols):
        idx = np.where(mask[:, c])[0]
        if idx.size == 0:
            continue
        row = float(idx.mean()) / float(max(rows - 1, 1))
        ys.append(row)
        xs.append((c + 0.5) / float(cols))
    if not xs:
        return None
    return xs, ys
