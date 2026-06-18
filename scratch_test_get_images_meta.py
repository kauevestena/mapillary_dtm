import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from dtm_from_mapillary.api.mapillary_client import MapillaryClient
c = MapillaryClient()
bbox = [-48.596644, -27.591363, -48.589890, -27.586780]
imgs = c.list_images_in_bbox(bbox, limit=100)
img_ids = [i["id"] for i in imgs[:4]]
print("IDs:", img_ids)
detailed = []
for iid in img_ids:
    detailed.append(c.get_image_meta(iid, fields=["id", "thumb_1024_url"]))
print("Detailed:", detailed)
