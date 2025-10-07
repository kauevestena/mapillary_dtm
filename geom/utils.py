"""Shared helpers for lightweight reconstruction scaffolding."""
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


def heading_matrix(positions: np.ndarray, idx: int) -> np.ndarray:
    forward = _forward_vector(positions, idx)
    yaw = math.atan2(forward[1], forward[0])
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def synthetic_ground_offsets() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, -1.6],
            [1.2, 0.5, -1.5],
            [-1.2, 0.4, -1.55],
        ],
        dtype=float,
    )


def _forward_vector(positions: np.ndarray, idx: int) -> np.ndarray:
    if positions.shape[0] <= 1:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    if idx < positions.shape[0] - 1:
        direction = positions[idx + 1] - positions[idx]
    else:
        direction = positions[idx] - positions[idx - 1]
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return direction / norm


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
