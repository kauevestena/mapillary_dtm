"""
Writers for LAZ/GeoTIFF and manifests.
"""
from __future__ import annotations
from typing import List, Dict, Tuple
import json, os, math, time
import numpy as np

def write_geotiffs(out_dir: str, dtm: np.ndarray, slope_deg: np.ndarray, confidence: np.ndarray, transform=None, crs="EPSG:4978"):
    """
    Write GeoTIFF rasters for DTM, slope (deg), and confidence.
    (Implementation placeholder using rasterio.)
    """
    import rasterio
    os.makedirs(out_dir, exist_ok=True)
    for name, arr in [("dtm_0p5m_ellipsoid.tif", dtm),
                      ("slope_deg.tif", slope_deg),
                      ("confidence.tif", confidence)]:
        path = os.path.join(out_dir, name)
        with rasterio.open(path, "w",
                           driver="GTiff",
                           height=arr.shape[0],
                           width=arr.shape[1],
                           count=1,
                           dtype=str(arr.dtype),
                           crs=crs,
                           transform=transform) as dst:
            dst.write(arr, 1)

def write_laz(out_dir: str, points: np.ndarray, attrs: Dict[str, np.ndarray] | None = None, crs_wkt: str | None = None):
    """
    Write ground points to LAZ with optional attributes.
    """
    import laspy
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "ground_points.laz")
    hdr = laspy.LasHeader(point_format=3, version="1.2")
    if crs_wkt:
        try:
            from laspy import VLR, ExtraBytesParams
            pass
        except Exception:
            pass
    las = laspy.LasData(hdr)
    las.x = points[:,0]
    las.y = points[:,1]
    las.z = points[:,2]
    if attrs:
        for k, v in attrs.items():
            try:
                las.add_extra_dim(laspy.ExtraBytesParams(name=k, type=v.dtype))
                las[k] = v
            except Exception:
                # Fallback: skip attribute if extra bytes can't be added
                pass
    las.write(path)
