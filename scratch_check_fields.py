import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from dtm_from_mapillary.api.mapillary_client import MapillaryClient
c = MapillaryClient()
bbox = [-48.596644, -27.591363, -48.589890, -27.586780]
imgs = c.list_images_in_bbox(bbox, limit=1)
print(imgs)
if imgs:
    detailed = c.get_image_meta(imgs[0]["id"], fields=["id", "thumb_256_url", "thumb_1024_url", "thumb_2048_url", "thumb_original_url"])
    print(detailed)
