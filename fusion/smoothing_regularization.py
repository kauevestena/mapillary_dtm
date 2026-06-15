"""
Edge-aware smoothing tuned for slope preservation.
"""
from __future__ import annotations

import math

import numpy as np

from .. import constants

def edge_aware(dtm: np.ndarray) -> np.ndarray:
    """Apply lightweight edge-aware smoothing to the DTM."""

    if dtm.size == 0:
        return dtm

    arr = np.asarray(dtm, dtype=np.float32).copy()
    mask = np.isfinite(arr)
    if not mask.any():
        return arr

    sigma = max(constants.SMOOTHING_SIGMA_M, 0.1) if hasattr(constants, "SMOOTHING_SIGMA_M") else 0.7

    for _ in range(2):
        arr = _smooth_iteration(arr, mask, sigma)

    return arr


def _smooth_iteration(arr: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    height, width = arr.shape
    inv_two_sigma2 = 1.0 / max(2.0 * sigma * sigma, 1e-6)

    pad_arr = np.pad(arr, pad_width=1, mode='edge')
    pad_mask = np.pad(mask, pad_width=1, mode='constant', constant_values=False)
    
    total = np.zeros_like(arr)
    weight_sum = np.zeros_like(arr)

    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            neigh_arr = pad_arr[1+dy:height+1+dy, 1+dx:width+1+dx]
            neigh_mask = pad_mask[1+dy:height+1+dy, 1+dx:width+1+dx]
            
            diff = neigh_arr - arr
            w = np.exp(-(diff * diff) * inv_two_sigma2)
            w = np.where(neigh_mask, w, 0.0)
            
            total += neigh_arr * w
            weight_sum += w

    out = np.where(mask & (weight_sum > 0), total / weight_sum, arr)
    return out
