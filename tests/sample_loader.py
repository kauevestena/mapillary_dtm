import json
from pathlib import Path
from dtm_from_mapillary.common_core import FrameMeta

def get_sample_frames():
    meta_path = Path(__file__).resolve().parents[1] / "qa" / "data" / "sample_dataset" / "metadata.json"
    if not meta_path.exists():
        raise RuntimeError(f"Sample dataset not found at {meta_path}")
        
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
        
    imagery_root = meta_path.parent
    return {seq_id: frames}, imagery_root
