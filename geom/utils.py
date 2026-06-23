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

def umeyama_alignment(src: np.ndarray, dst: np.ndarray, estimate_scale: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Computes the Umeyama similarity transform to align src to dst.
    Returns (R, t, s) such that dst ~ s * R @ src + t
    """
    num = src.shape[0]
    if num < 3:
        return np.eye(3), np.zeros(3), 1.0

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    A = dst_demean.T @ src_demean / num

    d_src = np.var(src, axis=0).sum()

    U, S, Vt = np.linalg.svd(A)
    R = U @ Vt
    
    # Handle reflection
    if np.linalg.det(R) < 0:
        S[-1] = -S[-1]
        U[:, -1] = -U[:, -1]
        R = U @ Vt

    # For 1D trajectories, SVD might arbitrarily flip the "Up" vector (Z-axis).
    # If the Z-axis is flipped (R[2, 2] < 0), we can rotate by 180 deg around the dominant axis
    # by negating the 2nd and 3rd columns of U.
    if R[2, 2] < 0:
        U[:, 1] = -U[:, 1]
        U[:, 2] = -U[:, 2]
        R = U @ Vt

    s = 1.0
    if estimate_scale and d_src > 1e-8:
        s = 1.0 / d_src * S.sum()

    t = dst_mean - s * R @ src_mean

    return R, t, s

