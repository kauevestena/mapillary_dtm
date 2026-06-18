import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from dtm_from_mapillary.common_core import FrameMeta
from dtm_from_mapillary.geom.vo_simplified import run as run_vo

meta_path = Path("qa/data/sample_dataset/metadata.json")
data = json.loads(meta_path.read_text())
seq_id = list(data.keys())[0]
frames_data = data[seq_id]

frames = []
for d in frames_data:
    frames.append(FrameMeta(
        image_id=d["image_id"],
        seq_id=d["seq_id"],
        captured_at_ms=d["captured_at_ms"],
        lon=d["lon"],
        lat=d["lat"],
        alt_ellip=10.0,
        camera_type=d["camera_type"],
        cam_params={"focal": d["camera_parameters"][0], "k1": d["camera_parameters"][1], "k2": d["camera_parameters"][2]} if d.get("camera_parameters") else {},
        quality_score=d.get("quality_score", 0.5),
        thumbnail_url=d.get("thumbnail_url"),
    ))

imagery_root = Path("qa/data/sample_dataset")

# debug imagery loading
from dtm_from_mapillary.ingest.image_loader import ImageryLoader
loader = ImageryLoader(imagery_root)
img0 = loader.load_gray(frames[0])
print(f"Loaded img0: {img0 is not None}")

try:
    results = run_vo({seq_id: frames}, imagery_root=imagery_root, min_inliers=10)
    print("VO SUCCESS:", results.keys())
    print("Metadata:", results[seq_id].metadata)
except Exception as e:
    print("VO FAILED:", e)

