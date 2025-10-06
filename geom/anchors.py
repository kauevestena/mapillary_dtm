"""
Map-feature / detection anchors (vertical objects) and footpoint triangulation.
"""
from __future__ import annotations
from typing import Dict, List, Tuple

def find_anchors(seqs: dict, token: str | None = None) -> list[dict]:
    """
    Use Mapillary detections/map-features to collect candidate poles/signs/lights,
    detect their footpoints in images, and triangulate stable 3D anchor points.
    """
    # Placeholder
    return []
