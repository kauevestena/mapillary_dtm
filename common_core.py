"""
Shared dataclasses and core math utilities.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np

@dataclass
class FrameMeta:
    image_id: str
    seq_id: str
    captured_at_ms: int
    lon: float
    lat: float
    alt_ellip: Optional[float]
    camera_type: str  # "perspective"|"fisheye"|"spherical"
    cam_params: Dict  # fx, fy, cx, cy, distortion, etc (OpenSfM-like)
    quality_score: Optional[float]

@dataclass
class Pose:
    """World-from-camera pose (R,t). Coordinates are meters in a local ENU frame."""
    R: np.ndarray  # (3,3)
    t: np.ndarray  # (3,)

@dataclass
class GroundPoint:
    x: float; y: float; z: float
    method: str                 # "opensfm"|"colmap"|"vo+mono"|"anchor"
    seq_id: str
    image_ids: List[str]
    view_count: int
    sem_prob: float
    tri_angle_deg: Optional[float]
    uncertainty_m: float

def wgs84_to_enu(lon: float, lat: float, h: float,
                 lon0: float, lat0: float, h0: float) -> np.ndarray:
    """
    Convert WGS84 (lon,lat,h) to local ENU given origin (lon0,lat0,h0).
    Requires pyproj at runtime.
    """
    try:
        from pyproj import Transformer
    except ImportError as e:
        raise RuntimeError("pyproj is required for wgs84_to_enu") from e
    t_ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    x, y, z = t_ecef.transform(lon, lat, h)
    x0, y0, z0 = t_ecef.transform(lon0, lat0, h0)
    # Build ENU transform
    import math
    lam = math.radians(lon0); phi = math.radians(lat0)
    sl, cl = math.sin(lam), math.cos(lam)
    sp, cp = math.sin(phi), math.cos(phi)
    R = np.array([[-sl,          cl,           0],
                  [-sp*cl, -sp*sl,  cp],
                  [ cp*cl,  cp*sl,  sp]])
    enu = R @ np.array([x - x0, y - y0, z - z0])
    return enu

def enu_to_wgs84(x: float, y: float, z: float,
                 lon0: float, lat0: float, h0: float) -> Tuple[float,float,float]:
    """Inverse of wgs84_to_enu (approximate)."""
    try:
        from pyproj import Transformer
    except ImportError as e:
        raise RuntimeError("pyproj is required for enu_to_wgs84") from e
    import math, numpy as np
    lam = math.radians(lon0); phi = math.radians(lat0)
    sl, cl = math.sin(lam), math.cos(lam)
    sp, cp = math.sin(phi), math.cos(phi)
    Rt = np.array([[-sl, -sp*cl,  cp*cl],
                   [ cl, -sp*sl,  cp*sl],
                   [  0,      cp,     sp]])
    dx, dy, dz = Rt @ np.array([x, y, z])
    t_ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    x0, y0, z0 = t_ecef.transform(lon0, lat0, h0)
    xe, ye, ze = x0 + dx, y0 + dy, z0 + dz
    t_llh = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)
    lon, lat, h = t_llh.transform(xe, ye, ze)
    return lon, lat, h

def camera_ray(u: float, v: float, cam_params: Dict, cam_type: str) -> np.ndarray:
    """
    Compute unit ray in camera frame for a pixel (u,v).
    - perspective: uses intrinsics (fx, fy, cx, cy)
    - fisheye/spherical: placeholder models; refine in implementation
    """
    if cam_type == "perspective":
        fx, fy = cam_params.get("fx"), cam_params.get("fy")
        cx, cy = cam_params.get("cx"), cam_params.get("cy")
        x = (u - cx) / fx
        y = (v - cy) / fy
        r = np.array([x, y, 1.0], dtype=float)
        return r / np.linalg.norm(r)
    elif cam_type in ("fisheye", "spherical"):
        # TODO: implement accurate fisheye/equirectangular ray models per OpenSfM
        raise NotImplementedError("camera_ray for fisheye/spherical not implemented yet")
    else:
        raise ValueError(f"Unknown camera_type: {cam_type}")

def ray_plane_intersect(C: np.ndarray, r: np.ndarray,
                        plane_n: np.ndarray, plane_p: np.ndarray) -> float:
    """
    Return lambda* such that C + lambda*r intersects plane with normal n at point p.
    """
    denom = float(np.dot(plane_n, r))
    if abs(denom) < 1e-9:
        return np.inf
    lam = float(np.dot(plane_n, plane_p - C) / denom)
    return lam

def fit_plane_ransac(P: np.ndarray, iters: int = 200, tau: float = 0.05) -> tuple[np.ndarray,np.ndarray]:
    """
    Fit plane n^T(x - p0) = 0 using RANSAC. Returns (n, p0).
    """
    rng = np.random.default_rng(0)
    best_inl, best_model = [], (np.array([0,0,1.0]), np.zeros(3))
    N = P.shape[0]
    if N < 3: return best_model
    for _ in range(iters):
        idx = rng.choice(N, 3, replace=False)
        A = P[idx]
        n = np.cross(A[1]-A[0], A[2]-A[0])
        if np.linalg.norm(n) < 1e-9: continue
        n = n / np.linalg.norm(n)
        d = np.abs((P - A[0]) @ n)
        inl = np.where(d < tau)[0]
        if len(inl) > len(best_inl):
            best_inl = inl.tolist()
            best_model = (n, A[0])
    return best_model

def percentile_lower(values, q: float = 0.25):
    a = np.asarray(values, dtype=float)
    if a.size == 0: return np.nan
    return float(np.nanpercentile(a, q*100.0))

def robust_mean(values, w=None):
    a = np.asarray(values, dtype=float)
    if w is None: w = np.ones_like(a)
    if a.size == 0: return np.nan
    return float(np.sum(a*w) / max(np.sum(w), 1e-9))

def tile_bounds(bbox, tile_size_m):
    """
    Break a bbox into ENU tiles of approximately tile_size_m. Placeholder for scheduler.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    return [dict(bbox=bbox, idx=0)]
