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


def write_ply_from_geotiff(tiff_path: str, output_dir: str) -> str:
    """
    Convert a GeoTIFF DTM file to PLY point cloud format.
    
    Args:
        tiff_path (str): Path to the input GeoTIFF file
        output_dir (str): Directory where output files will be saved
        
    Returns:
        str: Path to the created PLY file
    """
    import rasterio
    from pathlib import Path
    
    # Read the GeoTIFF file
    with rasterio.open(tiff_path) as src:
        # Read the elevation data
        dtm = src.read(1)
        
        # Get geospatial information
        transform = src.transform
        crs = src.crs
        
        # Get image dimensions
        height, width = dtm.shape
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Convert to point cloud (PLY format)
        xyz_points = []
        
        # Iterate through the grid and create 3D points
        for y in range(height):
            for x in range(width):
                # Skip no-data values (typically -9999 or NaN)
                if dtm[y, x] < -1000 or np.isnan(dtm[y, x]):
                    continue
                
                # Convert pixel coordinates to geographic coordinates
                # Using the affine transform to get world coordinates
                lon, lat = transform * (x, y)
                
                # Create point (x, y, z) where z is the elevation
                point = [lon, lat, dtm[y, x]]
                xyz_points.append(point)
        
        # Write PLY file (ASCII format)
        ply_path = Path(output_dir) / "dtm_0p5m_ellipsoid.ply"
        
        with open(ply_path, "w") as f:
            # PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(xyz_points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            
            # Write points
            for point in xyz_points:
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
    
    return str(ply_path)
