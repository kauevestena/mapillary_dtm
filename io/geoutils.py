"""
Geospatial utilities: CRS transforms, geoid correction, ENU frames.
"""
from __future__ import annotations
from typing import Tuple, Optional
from ..common_core import wgs84_to_enu, enu_to_wgs84

def apply_geoid_correction(h_ellip: float, geoid_model: str = "EGM2008") -> float:
    """
    Placeholder: return orthometric height H = h - N(geoid). To be implemented with geographiclib.
    """
    raise NotImplementedError("Geoid correction not implemented; ellipsoidal heights are the default.")
