import sys
from pathlib import Path

# Add the my_mapillary_api directory to Python path to import it directly
sys.path.insert(0, str(Path(__file__).parent / "api" / "my_mapillary_api"))

from mapillary_api import *


# read the token from
with open("mapillary_token", "r") as f:
    token = f.read().strip()


from constants import *

# Query images in a bounding box (minLon, minLat, maxLon, maxLat)
metadata = get_mapillary_images_metadata(
    bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"], token=token
)

# Convert to GeoDataFrame
gdf = mapillary_data_to_gdf(metadata)

print(gdf)
