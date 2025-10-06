"""
Internal QA: slopes, agreement maps, elevated-structure masking.
"""
from __future__ import annotations
import numpy as np
from .. import constants

def slope_from_plane_fit(dtm: np.ndarray, win: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute slope (deg) and aspect via local plane fits.
    """
    win = win or constants.SLOPE_FROM_FIT_SIZE
    H, W = dtm.shape
    slope = np.full_like(dtm, np.nan, dtype=np.float32)
    aspect = np.full_like(dtm, np.nan, dtype=np.float32)
    # Placeholder: central differences fallback
    gy, gx = np.gradient(dtm.astype(np.float32))
    slope_rad = np.arctan(np.hypot(gx, gy))
    slope[:] = np.degrees(slope_rad)
    aspect[:] = (np.degrees(np.arctan2(-gx, gy)) + 360.0) % 360.0
    return slope, aspect

def write_agreement_maps(*args, **kwargs):
    pass
