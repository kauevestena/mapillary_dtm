# Mapillary API Module

This module provides two approaches for interacting with the Mapillary API:

## mapillary_client.py

Custom wrapper around the Mapillary Graph API v4. This is the primary client used throughout the codebase for:
- Fetching image metadata
- Downloading thumbnails
- Querying sequences
- Vector tile operations

## my_mapillary_api.py

Alternative Mapillary API helper from [kauevestena/my_mappilary_api](https://github.com/kauevestena/my_mappilary_api.git).

This module provides a GeoDataFrame-centric approach to working with Mapillary data:
- Direct GeoDataFrame conversion
- Tiled querying for large areas
- Batch image downloads
- Polygon-based filtering

### Source

- **Repository**: https://github.com/kauevestena/my_mappilary_api.git
- **License**: See LICENSE file in the source repository

### Dependencies

The custom API requires the following packages (all included in `requirements.txt`):
- `geopandas` - Geospatial data handling
- `requests` - HTTP requests
- `wget` - File downloads
- `mercantile` - Map tile utilities
- `tqdm` - Progress bars
- `pandas` - Data manipulation
- `shapely` - Geometric operations

## Migration from Official SDK

The official `mapillary` Python SDK has been replaced with the custom API implementations above. The codebase never directly imported from the official SDK, instead using the custom `MapillaryClient` wrapper which makes direct HTTP calls to the Mapillary Graph API v4.

### What Changed

1. Removed `mapillary` package dependency from `requirements.txt`
2. Added `wget` dependency (required by `my_mapillary_api.py`)
3. Integrated `my_mapillary_api.py` from the custom repository
4. Added this documentation

### No Code Changes Required

Since the official SDK was never used in the codebase, no imports or function calls needed to be updated. All existing code using `MapillaryClient` continues to work as before.
