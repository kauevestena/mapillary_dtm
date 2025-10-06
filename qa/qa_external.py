"""
External QA against held-out official GeoTIFFs.
"""
from __future__ import annotations
import numpy as np
from ..io.readers import read_raster

def compare_to_geotiff(dtm_path: str, check_path: str) -> dict:
    # Placeholder comparison: compute RMSE on overlapping nodata-masked area
    return {"rmse_z": None, "n": 0}
