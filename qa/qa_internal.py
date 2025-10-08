"""
Internal QA: slopes, agreement maps, elevated-structure masking.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Sequence

import numpy as np
from scipy.signal import correlate2d

from .. import constants


def slope_from_plane_fit(dtm: np.ndarray, win: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Compute slope (deg) and aspect via local plane fits."""

    if dtm.ndim != 2:
        raise ValueError("dtm must be a 2-D array")

    win = int(win or constants.SLOPE_FROM_FIT_SIZE)
    if win < 3:
        raise ValueError("window size must be >= 3")
    if win % 2 == 0:
        win += 1

    grid_res = float(constants.GRID_RES_M)
    dtm = np.asarray(dtm, dtype=np.float64)
    mask = np.isfinite(dtm)
    if not mask.any():
        nan = np.full(dtm.shape, np.nan, dtype=np.float32)
        return nan, nan

    valid = mask.astype(np.float64)
    filled = np.where(mask, dtm, 0.0)

    half = win // 2
    offsets = np.arange(-half, half + 1, dtype=np.float64) * grid_res
    dy_kernel = np.repeat(offsets[:, None], win, axis=1)
    dx_kernel = np.repeat(offsets[None, :], win, axis=0)
    ones_kernel = np.ones((win, win), dtype=np.float64)

    dx2_kernel = dx_kernel**2
    dy2_kernel = dy_kernel**2
    dxdy_kernel = dx_kernel * dy_kernel

    def _corr(arr, kernel):
        return correlate2d(arr, kernel, mode="same", boundary="symm")

    N = _corr(valid, ones_kernel)
    Sx = _corr(valid, dx_kernel)
    Sy = _corr(valid, dy_kernel)
    Sxx = _corr(valid, dx2_kernel)
    Syy = _corr(valid, dy2_kernel)
    Sxy = _corr(valid, dxdy_kernel)

    Sz = _corr(filled, ones_kernel)
    Sxz = _corr(filled, dx_kernel)
    Syz = _corr(filled, dy_kernel)

    A00 = Sxx
    A01 = Sxy
    A02 = Sx
    A11 = Syy
    A12 = Sy
    A22 = N

    b0 = Sxz
    b1 = Syz
    b2 = Sz

    detA = (
        A00 * (A11 * A22 - A12 * A12)
        - A01 * (A01 * A22 - A12 * A02)
        + A02 * (A01 * A12 - A11 * A02)
    )

    detAx = (
        b0 * (A11 * A22 - A12 * A12)
        - A01 * (b1 * A22 - A12 * b2)
        + A02 * (b1 * A12 - A11 * b2)
    )

    detAy = (
        A00 * (b1 * A22 - A12 * b2)
        - b0 * (A01 * A22 - A12 * A02)
        + A02 * (A01 * b2 - b1 * A02)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        grad_x = detAx / detA
        grad_y = detAy / detA

    valid_mask = (N >= 3) & np.isfinite(grad_x) & np.isfinite(grad_y) & (np.abs(detA) > 1e-9)
    grad_x = np.where(valid_mask, grad_x, np.nan)
    grad_y = np.where(valid_mask, grad_y, np.nan)

    slope_rad = np.arctan(np.hypot(grad_x, grad_y))
    slope = np.degrees(slope_rad).astype(np.float32)
    aspect = (np.degrees(np.arctan2(-grad_x, grad_y)) + 360.0) % 360.0
    aspect = aspect.astype(np.float32)

    slope[~valid_mask] = np.nan
    aspect[~valid_mask] = np.nan

    return slope, aspect


def write_agreement_maps(
    out_path: Path | str | None,
    fused_dtm: np.ndarray,
    source_dtms: Mapping[str, np.ndarray],
    view_counts: Mapping[str, np.ndarray] | None = None,
    slope_window: int | None = None,
) -> Dict[str, np.ndarray]:
    """Compute (and optionally persist) per-cell agreement diagnostics."""

    fused = np.asarray(fused_dtm, dtype=np.float32)
    if fused.ndim != 2:
        raise ValueError("fused_dtm must be a 2-D array")

    slope_window = slope_window or constants.SLOPE_FROM_FIT_SIZE

    diffs = []
    slope_diffs = []
    stacked_counts = []

    fused_slope, _ = slope_from_plane_fit(fused, win=slope_window)

    for name, arr in source_dtms.items():
        src = np.asarray(arr, dtype=np.float32)
        if src.shape != fused.shape:
            raise ValueError(f"Source '{name}' shape {src.shape} does not match fused {fused.shape}")
        mask = np.isfinite(src) & np.isfinite(fused)
        if not mask.any():
            continue
        diff = np.full_like(fused, np.nan, dtype=np.float32)
        diff[mask] = src[mask] - fused[mask]
        diffs.append(diff)

        src_slope, _ = slope_from_plane_fit(src, win=slope_window)
        slope_diff = np.full_like(fused, np.nan, dtype=np.float32)
        slope_mask = np.isfinite(src_slope) & np.isfinite(fused_slope)
        slope_diff[slope_mask] = src_slope[slope_mask] - fused_slope[slope_mask]
        slope_diffs.append(slope_diff)

        if view_counts and name in view_counts:
            vc = np.asarray(view_counts[name], dtype=np.float32)
            if vc.shape == fused.shape:
                stacked_counts.append(np.where(np.isfinite(vc), vc, 0.0))

    if not diffs:
        results = {
            "dz_mean_abs": np.full_like(fused, np.nan, dtype=np.float32),
            "dz_rmse": np.full_like(fused, np.nan, dtype=np.float32),
            "dz_max_abs": np.full_like(fused, np.nan, dtype=np.float32),
            "slope_mean_abs": np.full_like(fused, np.nan, dtype=np.float32),
            "source_count": np.zeros_like(fused, dtype=np.float32),
        }
    else:
        diff_stack = np.stack(diffs, axis=0)
        slope_stack = np.stack(slope_diffs, axis=0)
        mask = np.isfinite(diff_stack)

        abs_diff = np.abs(diff_stack)
        dz_mean_abs = np.nanmean(abs_diff, axis=0)
        dz_rmse = np.sqrt(np.nanmean(diff_stack**2, axis=0))
        dz_max_abs = np.nanmax(abs_diff, axis=0)

        slope_abs = np.abs(slope_stack)
        slope_mean_abs = np.nanmean(slope_abs, axis=0)

        source_count = np.sum(mask, axis=0, dtype=np.float32)
        if stacked_counts:
            source_count += np.sum(np.stack(stacked_counts, axis=0), axis=0)

        results = {
            "dz_mean_abs": dz_mean_abs.astype(np.float32),
            "dz_rmse": dz_rmse.astype(np.float32),
            "dz_max_abs": dz_max_abs.astype(np.float32),
            "slope_mean_abs": slope_mean_abs.astype(np.float32),
            "source_count": source_count.astype(np.float32),
        }

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **results)

    return results
