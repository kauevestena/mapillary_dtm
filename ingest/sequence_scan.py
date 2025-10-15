"""Sequence discovery over a bounding box using the Mapillary API."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from ..api.mapillary_client import MapillaryClient
from ..common_core import FrameMeta
from .. import constants
from . import cache_utils

log = logging.getLogger(__name__)


def discover_sequences(
    aoi_bbox: Sequence[float],
    token: Optional[str] = None,
    client: Optional[MapillaryClient] = None,
    max_sequences: Optional[int] = None,
    max_images_per_sequence: Optional[int] = None,
    cache_dir: Optional[Path | str] = None,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> Dict[str, List[FrameMeta]]:
    """Discover Mapillary sequences and per-frame metadata inside *aoi_bbox*.

    Parameters
    ----------
    aoi_bbox:
        Bounding box as (lon_min, lat_min, lon_max, lat_max).
    token:
        Optional Mapillary access token. If omitted the client will look for
        an environment variable or `mapillary_token` file.
    client:
        Injected MapillaryClient for testing.
    max_sequences / max_images_per_sequence:
        Optional limits to guard against very large AOIs during early
        development.

    Returns
    -------
    dict
        Mapping of `sequence_id -> List[FrameMeta]`, sorted by capture time.
    """

    bbox = tuple(float(x) for x in aoi_bbox)
    if len(bbox) != 4:
        raise ValueError("aoi_bbox must provide four coordinates: lon_min, lat_min, lon_max, lat_max")
    lon_min, lat_min, lon_max, lat_max = bbox
    if lon_min >= lon_max or lat_min >= lat_max:
        raise ValueError(f"Invalid bbox ordering: {bbox}")

    client = client or MapillaryClient(token=token)

    cache_dir_path = cache_utils.metadata_cache_dir(cache_dir)

    seq_ids = list(client.list_sequence_ids_in_bbox(bbox))
    if max_sequences is not None:
        seq_ids = seq_ids[:max_sequences]
    log.info("Discovered %d candidate sequences in bbox", len(seq_ids))

    sequences: Dict[str, List[FrameMeta]] = {}
    for seq_id in seq_ids:
        cached = None
        cache_path = cache_dir_path / f"{seq_id}.jsonl"
        if use_cache and cache_path.exists() and not force_refresh:
            cached = _load_cache(cache_path)
            if cached:
                sequences[seq_id] = cached
                log.debug("Loaded sequence %s from cache", seq_id)
                continue

        image_ids = client.list_image_ids_in_sequence(seq_id)
        if not image_ids:
            continue
        if max_images_per_sequence is not None:
            image_ids = image_ids[:max_images_per_sequence]

        metas = client.get_images_meta(image_ids)
        frames: List[FrameMeta] = []
        for meta in metas:
            frame = _frame_meta_from_api(meta)
            if frame is None:
                continue
            if frame.seq_id != str(seq_id):
                continue
            if not _within_bbox(frame.lon, frame.lat, bbox):
                continue
            frames.append(frame)

        if not frames:
            continue

        frames.sort(key=lambda f: f.captured_at_ms)
        sequences[seq_id] = frames
        log.debug("Sequence %s → %d frames", seq_id, len(frames))

        _write_cache(cache_path, frames)
        cache_utils.enforce_quota(
            cache_dir_path, constants.MAPILLARY_METADATA_CACHE_MAX_GB
        )
        log.debug("Wrote cache for sequence %s to %s", seq_id, cache_path)

    log.info("Retained %d sequences after bbox filtering", len(sequences))
    return sequences


def _frame_meta_from_api(meta: Mapping) -> Optional[FrameMeta]:
    seq_id = meta.get("sequence_id") or meta.get("sequence")
    image_id = meta.get("id")
    geometry = meta.get("geometry") or {}
    coords = geometry.get("coordinates") if isinstance(geometry, Mapping) else None

    if not seq_id or not image_id or not coords or len(coords) < 2:
        return None

    lon, lat = float(coords[0]), float(coords[1])
    alt = float(coords[2]) if len(coords) >= 3 and coords[2] is not None else None

    captured_at_ms = _captured_at_to_ms(meta.get("captured_at"))
    camera_type = (meta.get("camera_type") or "").lower() or "unknown"
    quality_score = _safe_float(meta.get("quality_score"))

    cam_params = meta.get("camera_parameters") or {}
    if isinstance(cam_params, str):
        # Some API responses return camera_parameters as JSON string
        cam_params = _safe_json_loads(cam_params) or {}
    if not isinstance(cam_params, Mapping):
        cam_params = {}
    thumb_url = meta.get("thumb_1024_url") or meta.get("thumb_2048_url")

    return FrameMeta(
        image_id=str(image_id),
        seq_id=str(seq_id),
        captured_at_ms=captured_at_ms,
        lon=lon,
        lat=lat,
        alt_ellip=alt,
        camera_type=camera_type,
        cam_params=dict(cam_params),
        quality_score=quality_score,
        thumbnail_url=str(thumb_url) if thumb_url else None,
    )


def _captured_at_to_ms(raw) -> int:
    if raw is None:
        return 0
    if isinstance(raw, (int, float)):
        # Graph API may already return milliseconds
        return int(raw)
    if isinstance(raw, str) and raw:
        try:
            if raw.endswith("Z"):
                raw_dt = raw[:-1] + "+00:00"
            else:
                raw_dt = raw
            dt = datetime.fromisoformat(raw_dt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000.0)
        except ValueError:
            log.debug("Unable to parse captured_at %s", raw)
    return 0


def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_json_loads(raw: str):
    import json

    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def _within_bbox(lon: float, lat: float, bbox: Sequence[float]) -> bool:
    lon_min, lat_min, lon_max, lat_max = bbox
    return (lon_min <= lon <= lon_max) and (lat_min <= lat <= lat_max)


def _write_cache(path: Path, frames: Iterable[FrameMeta]) -> None:
    try:
        with path.open("w", encoding="utf8") as fh:
            for frame in frames:
                fh.write(json.dumps(frame.to_dict(), separators=(",", ":")) + "\n")
    except OSError as exc:
        log.warning("Failed to write cache %s: %s", path, exc)


def _load_cache(path: Path) -> List[FrameMeta]:
    frames: List[FrameMeta] = []
    try:
        with path.open("r", encoding="utf8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    frames.append(FrameMeta.from_dict(data))
                except Exception:
                    continue
    except OSError as exc:
        log.debug("Failed to read cache %s: %s", path, exc)
        return []
    return frames
