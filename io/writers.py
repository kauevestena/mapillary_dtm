"""
Writers for LAZ/GeoTIFF and manifests.
"""
from __future__ import annotations
from typing import Dict
import os
import numpy as np

def write_geotiffs(out_dir: str, dtm: np.ndarray, slope_deg: np.ndarray, confidence: np.ndarray, transform=None, crs="EPSG:4978") -> Dict[str, str]:
    """
    Write GeoTIFF rasters for DTM, slope (deg), and confidence.
    (Implementation placeholder using rasterio.)
    """
    os.makedirs(out_dir, exist_ok=True)
    outputs: Dict[str, str] = {}
    datasets = [
        ("dtm_0p5m_ellipsoid.tif", dtm),
        ("slope_deg.tif", slope_deg),
        ("confidence.tif", confidence),
    ]
    try:
        import rasterio
    except ImportError:
        for name, arr in datasets:
            path = os.path.join(out_dir, name.replace(".tif", ".npy"))
            np.save(path, arr, allow_pickle=False)
            outputs[name] = path
        return outputs

    for name, arr in datasets:
        path = os.path.join(out_dir, name)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=arr.shape[0],
            width=arr.shape[1],
            count=1,
            dtype=str(arr.dtype),
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(arr, 1)
        outputs[name] = path
    return outputs

def write_laz(out_dir: str, points: np.ndarray, attrs: Dict[str, np.ndarray] | None = None, crs_wkt: str | None = None) -> str:
    """
    Write ground points to LAZ with optional attributes.
    """
    os.makedirs(out_dir, exist_ok=True)
    base_path = os.path.join(out_dir, "ground_points")
    try:
        import laspy
    except ImportError:
        fallback = base_path + ".npz"
        payload = {"points": points.astype(np.float32)}
        if attrs:
            for key, val in attrs.items():
                payload[f"attr_{key}"] = np.asarray(val)
        np.savez_compressed(fallback, **payload)
        return fallback

    path = base_path + ".laz"
    hdr = laspy.LasHeader(point_format=3, version="1.2")
    if crs_wkt:
        try:
            hdr.parse_crs_wkt(crs_wkt)
        except Exception:
            pass
    las = laspy.LasData(hdr)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    if attrs:
        for k, v in attrs.items():
            arr = np.asarray(v)
            try:
                las.add_extra_dim(laspy.ExtraBytesParams(name=k, type=arr.dtype))
                las[k] = arr
            except Exception:
                continue
    las.write(path)
    return path
