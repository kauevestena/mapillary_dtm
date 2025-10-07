"""Synthetic anchor discovery using cached detections or heuristics."""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np

from ..common_core import Anchor, AnchorObservation, FrameMeta, enu_to_wgs84
from .utils import positions_from_frames

log = logging.getLogger(__name__)


def find_anchors(
    seqs: Mapping[str, List[FrameMeta]],
    token: str | None = None,
    cache_dir: Path | str = Path("cache/anchors"),
    sample_path: Path | str = Path("qa/data/sample_anchors.json"),
) -> List[Anchor]:
    """Return vertical anchors per sequence.

    Preference order:
    1. Sequence-specific cache (`cache/anchors/<seq>.json`).
    2. Sample QA dataset (if sequence appears there).
    3. Synthetic heuristic using GNSS trajectory to place poles along curb.
    """

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    sample_map = _load_sample(sample_path)
    collected: List[Anchor] = []

    for seq_id, frames in seqs.items():
        anchors = _load_cache(cache_dir, seq_id)
        if anchors:
            log.debug("Loaded %d anchors for %s from cache", len(anchors), seq_id)
        elif seq_id in sample_map:
            anchors = sample_map[seq_id]
            _write_cache(cache_dir, seq_id, anchors)
        else:
            anchors = _synthesize_anchors(seq_id, frames)
            if anchors:
                _write_cache(cache_dir, seq_id, anchors)

        collected.extend(anchors)

    return collected


def _load_sample(sample_path: Path | str) -> Dict[str, List[Anchor]]:
    path = Path(sample_path)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf8"))
    except json.JSONDecodeError as exc:
        log.warning("Failed to parse sample anchors %s: %s", path, exc)
        return {}

    anchors_raw = raw.get("anchors")
    if anchors_raw is None:
        return {}

    seq_map: Dict[str, List[Anchor]] = defaultdict(list)
    for item in anchors_raw:
        try:
            anchor = Anchor.from_dict(item)
        except Exception as exc:  # pragma: no cover
            log.debug("Skipping malformed anchor entry: %s", exc)
            continue
        seq_map[anchor.seq_id].append(anchor)
    return seq_map


def _load_cache(cache_dir: Path, seq_id: str) -> List[Anchor]:
    path = cache_dir / f"{seq_id}.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf8"))
    except json.JSONDecodeError:
        return []
    anchors = []
    for item in payload.get("anchors", []):
        try:
            anchors.append(Anchor.from_dict(item))
        except Exception:
            continue
    return anchors


def _write_cache(cache_dir: Path, seq_id: str, anchors: Iterable[Anchor]) -> None:
    path = cache_dir / f"{seq_id}.json"
    try:
        data = {"anchors": [a.to_dict() for a in anchors]}
        path.write_text(json.dumps(data, indent=2), encoding="utf8")
    except OSError as exc:  # pragma: no cover
        log.warning("Failed to write anchor cache %s: %s", path, exc)


def _synthesize_anchors(seq_id: str, frames: List[FrameMeta]) -> List[Anchor]:
    if not frames:
        return []
    positions, origin = positions_from_frames(frames)
    if positions.shape[0] == 0:
        return []

    rng = np.random.default_rng(abs(hash(seq_id)) & 0xFFFF)
    anchors: List[Anchor] = []
    sample_indices = np.linspace(0, len(frames) - 1, num=min(3, len(frames)), dtype=int)

    for idx, frame_idx in enumerate(sample_indices):
        base = positions[frame_idx]
        offset = np.array([
            rng.uniform(3.0, 6.0) * (1 if idx % 2 == 0 else -1),
            rng.uniform(-2.0, 2.0),
            0.0,
        ])
        pole_height = float(rng.uniform(3.5, 5.5))
        top_local = base + offset + np.array([0.0, 0.0, pole_height])
        lon, lat, alt = enu_to_wgs84(top_local[0], top_local[1], top_local[2], *origin)

        observations = _synthetic_observations(frames, rng)
        anchors.append(
            Anchor(
                seq_id=seq_id,
                anchor_id=f"{seq_id}-anchor-{idx}",
                lon=lon,
                lat=lat,
                alt_ellip=alt,
                height_m=pole_height,
                diameter_m=0.3,
                source="synthetic",
                observations=observations,
            )
        )
    return anchors


def _synthetic_observations(frames: List[FrameMeta], rng: np.random.Generator, max_obs: int = 4) -> List[AnchorObservation]:
    obs: List[AnchorObservation] = []
    sample_frames = frames[: max_obs]
    for frame in sample_frames:
        params = frame.cam_params or {}
        width = float(params.get("width") or params.get("image_width") or 4000)
        height = float(params.get("height") or params.get("image_height") or 3000)
        px = width * rng.uniform(0.45, 0.55)
        py = height * rng.uniform(0.2, 0.6)
        obs.append(AnchorObservation(image_id=frame.image_id, px=px, py=py, prob=0.9))
    return obs
