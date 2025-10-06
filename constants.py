"""
Centralized configuration knobs and thresholds.
Change values here to tune behavior without touching the modules.
"""

# Spatial resolution
GRID_RES_M: float = 0.5
TILE_SIZE_M: int = 512

# Corridor (OSM-based, via OSMnx)
CORRIDOR_HALF_W_M: float = 25.0     # buffer around streets for "in-corridor" region
MAX_TIN_EXTRAPOLATION_M: float = 5.0  # limit when filling outside corridor
INCLUDE_INNER_BLOCKS: bool = True   # holes inside corridor polygons are filled
OSM_HIGHWAYS = [
    "motorway","trunk","primary","secondary","tertiary",
    "unclassified","residential","service"
]

# Imagery & selection
MIN_SPEED_KMH: float = 40.0
MAX_SPEED_KMH: float = 120.0
ALLOW_CAMERA_TYPES = {"perspective", "fisheye", "spherical"}
QUALITY_SCORE_MIN: float = 0.2

# SfM / VO / BA
MIN_TRIANG_ANGLE_DEG: float = 2.0
RANSAC_THRESH_PX: float = 1.0
LO_WINDOW_N: int = 5

# Height & scale
H_MIN_M: float = 1.0
H_MAX_M: float = 3.0
SCALE_GNSS_WT: float = 0.3
SCALE_ANCHOR_WT: float = 0.7

# Ground masking
GROUND_PROB_MIN: float = 0.6
EXCLUDE_CLASSES = {"vehicle", "person", "bike"}

# Fusion / surface
LOWER_ENVELOPE_Q: float = 0.25
SMOOTHING_SIGMA_M: float = 0.7
SLOPE_FROM_FIT_SIZE: int = 5

# Consensus
DZ_MAX_M: float = 0.25
DSLOPE_MAX_DEG: float = 2.0
MIN_SUPPORT_VIEWS: int = 3

# Elevated structures
EXCLUDE_ELEVATED_STRUCTURES: bool = True
ELEVATED_METHOD: str = "auto"  # "auto" => parallax + OSM bridge/tunnel tags

# QA
CHECKPOINT_BUFFER_M: float = 2.0

# API
MAPILLARY_GRAPH_URL = "https://graph.mapillary.com"
MAPILLARY_TILES_URL = "https://tiles.mapillary.com/maps/vtp"
DEFAULT_FIELDS = [
    "id","sequence_id","geometry","captured_at","camera_type","camera_parameters","quality_score"
]
