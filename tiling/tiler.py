"""
AOI chunking, corridor masking, and work scheduling.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
from ..api.tiles import bbox_to_z14_tiles

def plan_tiles(aoi_bbox: tuple[float,float,float,float]) -> List[dict]:
    return [dict(zxy=zxy) for zxy in bbox_to_z14_tiles(aoi_bbox)]
