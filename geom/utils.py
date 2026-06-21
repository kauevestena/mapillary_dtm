"""Shared helpers for lightweight reconstruction."""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

from ..common_core import FrameMeta, wgs84_to_enu


def positions_from_frames(frames: Sequence[FrameMeta]) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    if not frames:
        return np.empty((0, 3), dtype=float), (0.0, 0.0, 0.0)

    origin = frames[0]
    lon0, lat0 = float(origin.lon), float(origin.lat)
    h0 = float(origin.alt_ellip or 0.0)

    positions = []
    for frame in frames:
        lon, lat = float(frame.lon), float(frame.lat)
        h = float(frame.alt_ellip) if frame.alt_ellip is not None else h0
        positions.append(_to_local(lon, lat, h, lon0, lat0, h0))
    return np.asarray(positions, dtype=float), (lon0, lat0, h0)


def _to_local(lon: float, lat: float, h: float, lon0: float, lat0: float, h0: float) -> np.ndarray:
    try:
        return wgs84_to_enu(lon, lat, h, lon0, lat0, h0)
    except RuntimeError:
        # Fallback: simple equirectangular approximation
        R = 6378137.0
        dlon = math.radians(lon - lon0)
        dlat = math.radians(lat - lat0)
        x = R * dlon * math.cos(math.radians((lat + lat0) * 0.5))
        y = R * dlat
        z = h - h0
        return np.array([x, y, z], dtype=float)
