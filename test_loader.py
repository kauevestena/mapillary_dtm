import sys
from pathlib import Path
from dtm_from_mapillary.ingest.image_loader import ImageryLoader
from dtm_from_mapillary.common_core import FrameMeta
import json

base = "./imagery_cache"
loader = ImageryLoader(base)

# Mock a frame
class MockFrame:
    def __init__(self, seq_id, image_id):
        self.seq_id = seq_id
        self.image_id = image_id
        self.camera_model = None

frame = MockFrame("c6ixrty4mug2hb0gebz9rm", "217954769818203")
print("Seq dir:", loader._sequence_dir(frame.seq_id))
print("Candidate paths:")
for p in loader._candidate_paths(loader._sequence_dir(frame.seq_id), frame.image_id):
    print("  ", p, p.exists())
img = loader.load_rgb(frame)
print("Image loaded:", img is not None)
