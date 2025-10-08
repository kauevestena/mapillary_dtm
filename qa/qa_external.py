"""
External QA against held-out official GeoTIFFs.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from ..io.readers import read_raster
from .qa_internal import slope_from_plane_fit


def compare_to_geotiff(dtm_path: str, check_path: str) -> Dict[str, float]:
    """Compare generated DTM against a reference GeoTIFF."""

    dtm_arr, dtm_transform, dtm_crs = read_raster(dtm_path)
    if dtm_arr.ndim != 2:
        raise ValueError("DTM raster must be single-band")

    with rasterio.open(check_path) as ref:
        ref_arr = ref.read(1)
        dest = np.full(dtm_arr.shape, np.nan, dtype=np.float32)

        reproject(
            source=ref_arr,
            destination=dest,
            src_transform=ref.transform,
            src_crs=ref.crs,
            src_nodata=ref.nodata,
            dst_transform=dtm_transform,
            dst_crs=dtm_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    dtm = np.asarray(dtm_arr, dtype=np.float32)
    ref_resampled = np.asarray(dest, dtype=np.float32)

    mask = np.isfinite(dtm) & np.isfinite(ref_resampled)
    n = int(mask.sum())
    if n == 0:
        return {
            "rmse_z": float("nan"),
            "bias_z": float("nan"),
            "mae_z": float("nan"),
            "rmse_slope_deg": float("nan"),
            "n": 0,
        }

    dz = dtm[mask] - ref_resampled[mask]
    rmse_z = float(np.sqrt(np.mean(dz**2)))
    bias_z = float(np.mean(dz))
    mae_z = float(np.mean(np.abs(dz)))

    slope_dtm, _ = slope_from_plane_fit(dtm)
    slope_ref, _ = slope_from_plane_fit(ref_resampled)
    slope_mask = np.isfinite(slope_dtm) & np.isfinite(slope_ref)
    slope_mask &= mask
    if slope_mask.any():
        slope_diff = slope_dtm[slope_mask] - slope_ref[slope_mask]
        rmse_slope = float(np.sqrt(np.mean(slope_diff**2)))
    else:
        rmse_slope = float("nan")

    return {
        "rmse_z": rmse_z,
        "bias_z": bias_z,
        "mae_z": mae_z,
        "rmse_slope_deg": rmse_slope,
        "n": n,
    }
