"""Acquisition and optional SfM backend settings.

Slope/quality policies live in the versioned slope project JSON, not here.
"""
import os

# Walking and cycling imagery are eligible; quality is measured by geometry.
MIN_SPEED_KMH = 0.0
MAX_SPEED_KMH = 150.0
ALLOW_CAMERA_TYPES = {"perspective", "fisheye", "spherical"}
QUALITY_SCORE_MIN = 0.2
COLMAP_DEFAULT_THREADS = 8
COLMAP_USE_GPU = os.getenv("COLMAP_USE_GPU", "").lower() in {"1", "true"}

# API
MAPILLARY_GRAPH_URL = "https://graph.mapillary.com"
MAPILLARY_TILES_URL = "https://tiles.mapillary.com/maps/vtp"
DEFAULT_FIELDS = [
    "id",
    "sequence_id",
    "geometry",
    "captured_at",
    "camera_type",
    "camera_parameters",
    "quality_score",
    "thumb_1024_url",
]

# Mapillary cache configuration
MAPILLARY_CACHE_ROOT = os.getenv(
    "MAPILLARY_CACHE_ROOT",
    os.path.join(os.getenv("DTM_CACHE_ROOT", "cache"), "mapillary"),
)
MAPILLARY_METADATA_CACHE_MAX_GB = 2.0
MAPILLARY_IMAGERY_CACHE_MAX_GB = 8.0
MAPILLARY_DEFAULT_IMAGE_RES = 1024

# ── Deep-Image-Matching (DIM) ───────────────────────────────────────
DIM_ENABLED: bool = True  # DIM is the default matching backend
DIM_EXTRACTOR: str = os.getenv("DIM_EXTRACTOR", "superpoint")  # superpoint | aliked | sift | orb
DIM_MATCHER: str = os.getenv("DIM_MATCHER", "lightglue")  # lightglue | superglue | nn
DIM_TILE_SIZE: int = 2048  # for high-resolution images
DIM_MAX_FEATURES: int = 8192
DIM_QUALITY: str = "high"  # low | medium | high | highest

# sample region bbox (for testing):
# min_lon, min_lat, max_lon, max_lat = -48.596644,-27.591363,-48.589890,-27.586780
bbox = {
    "min_lon": -48.596644,
    "min_lat": -27.591363,
    "max_lon": -48.589890,
    "max_lat": -27.586780,
}  # Florianópolis, SC, Brazil

