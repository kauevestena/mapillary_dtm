"""Filter Mapillary sequences to fast-moving (likely car) segments."""
from __future__ import annotations

import logging
from typing import Dict, List

try:
    from pyproj import Geod
except ImportError as exc:  # pragma: no cover - handled at runtime
    Geod = None
    _GEOD = None
    _GEOD_IMPORT_ERROR = exc
else:
    _GEOD = Geod(ellps="WGS84")
    _GEOD_IMPORT_ERROR = None

from .. import constants
from ..common_core import FrameMeta

log = logging.getLogger(__name__)


def filter_car_sequences(seqs: Dict[str, List[FrameMeta]]) -> Dict[str, List[FrameMeta]]:
    """Filter sequences by speed, camera type, and quality score."""

    if _GEOD is None:
        raise RuntimeError("pyproj is required for filter_car_sequences") from _GEOD_IMPORT_ERROR

    filtered: Dict[str, List[FrameMeta]] = {}
    for seq_id, frames in seqs.items():
        if not frames:
            continue

        ordered = sorted(frames, key=lambda f: f.captured_at_ms)
        speed_buckets = [[] for _ in ordered]

        for idx in range(len(ordered) - 1):
            f0, f1 = ordered[idx], ordered[idx + 1]
            dt_s = max((f1.captured_at_ms - f0.captured_at_ms) / 1000.0, 0.0)
            if dt_s <= 0.1:  # ignore stalled or duplicate timestamps
                continue
            try:
                _, _, dist_m = _GEOD.inv(f0.lon, f0.lat, f1.lon, f1.lat)
            except Exception:  # pragma: no cover - pyproj raises rarely
                continue
            speed_kmh = (dist_m / dt_s) * 3.6
            speed_buckets[idx].append(speed_kmh)
            speed_buckets[idx + 1].append(speed_kmh)

        kept_frames: List[FrameMeta] = []
        for idx, frame in enumerate(ordered):
            if frame.camera_type and frame.camera_type.lower() not in constants.ALLOW_CAMERA_TYPES:
                continue
            if frame.quality_score is not None and frame.quality_score < constants.QUALITY_SCORE_MIN:
                continue
            speed_samples = speed_buckets[idx]
            if not speed_samples:
                continue
            speed_kmh = sum(speed_samples) / len(speed_samples)
            if speed_kmh < constants.MIN_SPEED_KMH or speed_kmh > constants.MAX_SPEED_KMH:
                continue
            kept_frames.append(frame)

        if kept_frames:
            filtered[seq_id] = kept_frames

    log.info("Car filter reduced %d sequences to %d", len(seqs), len(filtered))
    return filtered
