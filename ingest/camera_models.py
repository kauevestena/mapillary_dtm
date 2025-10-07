"""
Build OpenSfM-compatible camera models from Mapillary metadata.
"""
from __future__ import annotations
from typing import Dict, Optional
from ..common_core import FrameMeta

_PROJECTION_MAP = {
    "perspective": "perspective",
    "brown": "brown",
    "fisheye": "fisheye",
    "omnidirectional": "fisheye",
    "spherical": "spherical",
    "equirectangular": "spherical",
}


def make_opensfm_model(frame: FrameMeta) -> Dict:
    """Convert Mapillary camera metadata into OpenSfM camera format."""

    cam_type = (frame.camera_type or "perspective").lower()
    model: Dict[str, float | int | str | list] = {
        "projection_type": _PROJECTION_MAP.get(cam_type, "perspective")
    }

    params = frame.cam_params or {}
    width = _safe_int(params.get("width") or params.get("image_width"))
    height = _safe_int(params.get("height") or params.get("image_height"))
    if width:
        model["width"] = width
    if height:
        model["height"] = height

    fx = _safe_float(params.get("fx") or params.get("focal_x") or params.get("focal"))
    fy = _safe_float(params.get("fy") or params.get("focal_y") or params.get("focal"))
    focal_norm = _normalize_focal(fx, width)
    if focal_norm is not None:
        model["focal"] = focal_norm
    focal_y_norm = _normalize_focal(fy, width)
    if focal_y_norm is not None:
        model["focal_y"] = focal_y_norm

    principal = _principal_point(params, width, height)
    if principal:
        model["principal_point"] = principal

    for key in ("k1", "k2", "k3", "p1", "p2", "k4", "k5", "k6"):
        val = params.get(key)
        fval = _safe_float(val)
        if fval is not None:
            model[key] = fval

    if "skew" in params:
        skew = _safe_float(params.get("skew"))
        if skew is not None:
            model["skew"] = skew

    return model


def _normalize_focal(value: Optional[float], width: Optional[int]) -> Optional[float]:
    if value is None or not width:
        return None
    if width <= 0:
        return None
    return float(value) / float(width)


def _principal_point(params: Dict, width: Optional[int], height: Optional[int]) -> Optional[list]:
    cx = params.get("cx") or params.get("principal_x")
    cy = params.get("cy") or params.get("principal_y")
    px = _safe_float(cx)
    py = _safe_float(cy)
    if px is None or py is None or not width or not height or width <= 0 or height <= 0:
        return None
    return [px / float(width), py / float(height)]


def _safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
