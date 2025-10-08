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
    out = arr.copy()
    height, width = arr.shape
    inv_two_sigma2 = 1.0 / max(2.0 * sigma * sigma, 1e-6)

    for y in range(height):
        for x in range(width):
            if not mask[y, x]:
                continue
            base = float(arr[y, x])
            total = base
            weight_sum = 1.0

            y0 = max(0, y - 1)
            y1 = min(height - 1, y + 1)
            x0 = max(0, x - 1)
            x1 = min(width - 1, x + 1)

            for ny in range(y0, y1 + 1):
                for nx in range(x0, x1 + 1):
                    if nx == x and ny == y:
                        continue
                    if not mask[ny, nx]:
                        continue
                    diff = float(arr[ny, nx] - base)
                    w = math.exp(-(diff * diff) * inv_two_sigma2)
                    total += float(arr[ny, nx]) * w
                    weight_sum += w

            out[y, x] = float(total / weight_sum)

    return out
