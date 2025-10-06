"""
OSMnx helpers to derive corridor polygons (street vicinity).
"""
from __future__ import annotations
from typing import Tuple, List
from .. import constants

def corridor_from_osm_bbox(bbox: tuple[float,float,float,float]):
    """
    Use OSMnx to download roads in bbox, buffer by CORRIDOR_HALF_W_M, and
    build corridor polygons. Always include inner blocks (holes) inside corridor.
    """
    # Placeholder: implement with osmnx + shapely
    return None
