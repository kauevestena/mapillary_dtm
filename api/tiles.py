"""
Tile helpers for discovery via vector tiles.
"""
from __future__ import annotations
from typing import List, Tuple
import mercantile

def bbox_to_z14_tiles(bbox: tuple[float,float,float,float]) -> list[tuple[int,int,int]]:
    lon_min, lat_min, lon_max, lat_max = bbox
    tiles = mercantile.tiles(lon_min, lat_min, lon_max, lat_max, zooms=[14])
    return [(t.z, t.x, t.y) for t in tiles]
