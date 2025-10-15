# Helpers for caching Mapillary imagery locally.
from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

from .. import constants
from ..common_core import FrameMeta
from ..api.mapillary_client import MapillaryClient
from . import cache_utils

log = logging.getLogger(__name__)


def _resolve_resolution(resolution: int | None) -> int:
    if resolution is None or resolution <= 0:
        return constants.MAPILLARY_DEFAULT_IMAGE_RES
    return resolution


def prefetch_imagery(
    seqs: Mapping[str, Sequence[FrameMeta]],
    client: MapillaryClient,
    cache_dir: Path | str | None = None,
    max_per_sequence: int | None = 5,
    resolution: int | None = None,
    force_refresh: bool = False,
) -> MutableMapping[str, int]:
    """
    Download and cache Mapillary thumbnails for the provided sequences.

    Returns a mapping of `sequence_id -> images_cached`.
    """
    stats: MutableMapping[str, int] = {}
    if not seqs:
        return stats

    res = _resolve_resolution(resolution)
    imagery_root = cache_utils.imagery_cache_dir(cache_dir)

    for seq_id, frames in seqs.items():
        if not frames:
            continue
        seq_dir = cache_utils.sequence_imagery_dir(seq_id, base=cache_dir)
        cached_count = 0
        attempted = 0
        for frame in frames:
            if max_per_sequence is not None and attempted >= max_per_sequence:
                break
            attempted += 1

            dest_path = seq_dir / f"{frame.image_id}_{res}.jpg"
            if dest_path.exists() and not force_refresh:
                continue

            url = frame.thumbnail_url
            if not url:
                try:
                    url = client.get_thumbnail_url(frame.image_id, resolution=res)
                except Exception as exc:  # pragma: no cover - defensive
                    log.warning("Failed to request thumbnail URL for %s: %s", frame.image_id, exc)
                    continue
            if not url:
                continue

            try:
                client.download_file(url, dest_path)
                cached_count += 1
            except Exception as exc:  # pragma: no cover - network/IO errors
                log.warning("Failed to cache imagery for %s: %s", frame.image_id, exc)

        if cached_count:
            stats[seq_id] = cached_count

    cache_utils.enforce_quota(imagery_root, constants.MAPILLARY_IMAGERY_CACHE_MAX_GB)
    return stats
