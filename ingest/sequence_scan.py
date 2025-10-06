"""
Discover sequences and images in an AOI.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
from ..api.mapillary_client import MapillaryClient
from ..common_core import FrameMeta

def discover_sequences(aoi_bbox: tuple[float,float,float,float], token: str | None = None) -> dict[str, list[FrameMeta]]:
    """
    Return a mapping seq_id -> list[FrameMeta].
    Implementation placeholder: call MapillaryClient vector tiles + Graph paging.
    """
    client = MapillaryClient(token=token)
    # TODO: fetch sequences from vector tiles, then list image ids, then per-image meta
    return {}
