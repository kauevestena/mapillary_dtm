"""
Build OpenSfM-compatible camera models from Mapillary metadata.
"""
from __future__ import annotations
from typing import Dict
from ..common_core import FrameMeta

def make_opensfm_model(frame: FrameMeta) -> Dict:
    """
    Return a dict suitable for OpenSfM camera.json.
    """
    m = frame.cam_params.copy()
    m["projection_type"] = frame.camera_type  # "perspective" | "fisheye" | "spherical"
    return m
