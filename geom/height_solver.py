"""Per-sequence scale and camera-height estimation."""
from __future__ import annotations

import numpy as np
from collections import defaultdict
from typing import Dict, Iterable, Mapping, Tuple

from .. import constants
from ..common_core import Anchor, FrameMeta, ReconstructionResult
from .utils import positions_from_frames


def solve_scale_and_h(
    reconA: Mapping[str, ReconstructionResult],
    reconB: Mapping[str, ReconstructionResult],
    vo: Mapping[str, ReconstructionResult],
    anchors: Iterable[Anchor],
    seqs: Mapping[str, Iterable[FrameMeta]],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute per-sequence scale factors and camera heights."""

    anchors_by_seq = defaultdict(list)
    for anchor in anchors:
        anchors_by_seq[anchor.seq_id].append(anchor)

    scales: Dict[str, float] = {}
    heights: Dict[str, float] = {}

    for seq_id, frames in seqs.items():
        frames_list = list(frames)
        if not frames_list:
            continue

        gnss_avg = _average_step_from_gnss(frames_list)

        scale_candidates = []
        for recon_map in (reconA, reconB, vo):
            recon = recon_map.get(seq_id)
            if not recon:
                continue
            step = _average_step_from_recon(recon)
            if step > 1e-6 and gnss_avg > 1e-6:
                scale_candidates.append(gnss_avg / step)

        scale = float(np.mean(scale_candidates)) if scale_candidates else 1.0
        scale = max(0.25, min(4.0, scale))
        scales[seq_id] = scale

        anchors_list = anchors_by_seq.get(seq_id, [])
        height = _height_from_anchors(frames_list, anchors_list)
        heights[seq_id] = max(constants.H_MIN_M, min(constants.H_MAX_M, height))

    return scales, heights


def _average_step_from_gnss(frames: Iterable[FrameMeta]) -> float:
    frames = list(frames)
    if len(frames) < 2:
        return 1.0
    positions, _ = positions_from_frames(frames)
    if positions.shape[0] < 2:
        return 1.0
    diffs = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    if diffs.size == 0:
        return 1.0
    return float(np.mean(diffs))


def _average_step_from_recon(recon: ReconstructionResult) -> float:
    frames = recon.frames
    if len(frames) < 2:
        return 1.0
    translations = np.array([recon.poses[f.image_id].t for f in frames])
    diffs = np.linalg.norm(np.diff(translations, axis=0), axis=1)
    return float(np.mean(diffs)) if diffs.size else 1.0


def _height_from_anchors(frames: Iterable[FrameMeta], anchors: Iterable[Anchor]) -> float:
    frames = [f for f in frames if f.alt_ellip is not None]
    if not frames:
        return (constants.H_MIN_M + constants.H_MAX_M) * 0.5

    avg_cam_alt = float(np.mean([f.alt_ellip for f in frames]))
    base_alts = []
    for anchor in anchors:
        base_alt = anchor.alt_ellip - anchor.height_m
        base_alts.append(base_alt)
    if not base_alts:
        return (constants.H_MIN_M + constants.H_MAX_M) * 0.5
    avg_base_alt = float(np.mean(base_alts))
    height = avg_cam_alt - avg_base_alt
    if not np.isfinite(height):
        return (constants.H_MIN_M + constants.H_MAX_M) * 0.5
    return height
