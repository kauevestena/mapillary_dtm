"""
Readers for checkpoints and ancillary rasters.
"""
from __future__ import annotations
import rasterio
import numpy as np

def read_raster(path: str) -> tuple[np.ndarray, any, any]:
    with rasterio.open(path) as src:
        arr = src.read(1)
        return arr, src.transform, src.crs
