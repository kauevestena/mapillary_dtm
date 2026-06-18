#!/usr/bin/env python3
import sys
import logging
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dtm_from_mapillary.api.mapillary_client import MapillaryClient
from dtm_from_mapillary.ingest.sequence_scan import discover_sequences, _write_cache
from dtm_from_mapillary.ingest.imagery_cache import prefetch_imagery

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("fetch_sample")

def main():
    token_file = ROOT / "mapillary_token"
    token = token_file.read_text().strip() if token_file.exists() else None
    client = MapillaryClient(token=token)
    bbox = [-48.596644, -27.591363, -48.589890, -27.586780]
    
    log.info(f"Querying Mapillary API for sequences in bbox {bbox}...")
    
    # Use pipeline function to discover sequences
    # We disable cache writing temporarily to not pollute the QA folder with 100s of sequence metadata files
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        seqs = discover_sequences(bbox, client=client, cache_dir=td, use_cache=False)
    
    target_seq = None
    target_images = []
    
    for seq_id, frames in seqs.items():
        # filter for perspective cameras as done previously
        allowed_frames = [f for f in frames if f.camera_type in {"perspective", "fisheye", "spherical"}]
        if len(allowed_frames) >= 4:
            target_seq = seq_id
            target_images = allowed_frames[len(allowed_frames)//2 : len(allowed_frames)//2 + 4]
            break
            
    if not target_seq:
        log.error("Could not find a sequence with >= 4 allowed images in bbox.")
        sys.exit(1)
        
    log.info(f"Selected sequence {target_seq} with {len(target_images)} images.")
    
    out_dir = Path("qa/data/sample_dataset")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # We save exactly 4 frames into a single sequence mapping
    target_mapping = {target_seq: target_images}
    
    # Use pipeline function to download imagery
    log.info("Prefetching imagery using pipeline cache function...")
    prefetch_imagery(target_mapping, client=client, cache_dir=out_dir, resolution=1024)
    
    # We maintain the legacy metadata.json format since it's an explicit QA fixture
    saved_meta = []
    for frame in target_images:
        saved_meta.append({
            "image_id": frame.image_id,
            "seq_id": frame.seq_id,
            "captured_at_ms": frame.captured_at_ms,
            "lon": frame.lon,
            "lat": frame.lat,
            "camera_type": frame.camera_type,
            "camera_parameters": [
                frame.cam_params.get("focal", 0.0),
                frame.cam_params.get("k1", 0.0),
                frame.cam_params.get("k2", 0.0)
            ],
            "quality_score": frame.quality_score,
            "thumbnail_url": frame.thumbnail_url,
            "local_path": f"qa/data/sample_dataset/imagery/{target_seq}/{frame.image_id}_1024.jpg"
        })
        
    with open(out_dir / "metadata.json", "w") as f:
        json.dump({target_seq: saved_meta}, f, indent=2)
        
    log.info(f"Done! Saved metadata to {out_dir / 'metadata.json'}")

if __name__ == "__main__":
    main()
