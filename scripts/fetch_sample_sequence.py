#!/usr/bin/env python3
import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dtm_from_mapillary.api.mapillary_client import MapillaryClient

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("fetch_sample")

def main():
    client = MapillaryClient()
    bbox = [-48.596644, -27.591363, -48.589890, -27.586780]
    
    log.info(f"Querying Mapillary API for images in bbox {bbox}...")
    images = client.list_images_in_bbox(bbox, limit=1000)
    
    allowed = {"perspective", "fisheye", "spherical"}
    valid = [i for i in images if i.get("camera_type") in allowed]
    
    from collections import defaultdict
    seqs = defaultdict(list)
    for i in valid:
        seq = i.get("sequence")
        seq_id = None
        if isinstance(seq, dict):
            seq_id = seq.get("id")
        elif isinstance(seq, str):
            seq_id = seq
        else:
            seq_id = i.get("sequence_id")
            
        if seq_id:
            seqs[str(seq_id)].append(i)
            
    target_seq = None
    target_images = []
    
    for seq_id, imgs in seqs.items():
        if len(imgs) >= 4:
            imgs.sort(key=lambda x: x.get("captured_at", 0))
            target_seq = seq_id
            target_images = imgs[len(imgs)//2 : len(imgs)//2 + 4]
            break
            
    if not target_seq:
        log.error("Could not find a sequence with >= 4 allowed images in bbox.")
        sys.exit(1)
        
    log.info(f"Selected sequence {target_seq} with {len(target_images)} images.")
    
    img_ids = [i["id"] for i in target_images]
    detailed = []
    for iid in img_ids:
        detailed.append(client.get_image_meta(iid, fields=["id", "sequence", "geometry", "captured_at", "camera_type", "camera_parameters", "quality_score", "thumb_1024_url"]))
    
    out_dir = Path("qa/data/sample_dataset")
    img_dir = out_dir / "imagery" / target_seq
    img_dir.mkdir(parents=True, exist_ok=True)
    
    saved_meta = []
    for d in detailed:
        url = d.get("thumb_1024_url")
        if not url:
            log.warning(f"No 1024px thumb for {d['id']}, skipping")
            continue
            
        dest = img_dir / f"{d['id']}_1024.jpg"
        log.info(f"Downloading {dest} ...")
        client.download_file(url, dest)
        
        saved_meta.append({
            "image_id": d["id"],
            "seq_id": target_seq,
            "captured_at_ms": d["captured_at"],
            "lon": d["geometry"]["coordinates"][0],
            "lat": d["geometry"]["coordinates"][1],
            "camera_type": d.get("camera_type", "perspective"),
            "camera_parameters": d.get("camera_parameters", []),
            "quality_score": d.get("quality_score", 0.5),
            "thumbnail_url": url,
            "local_path": str(dest)
        })
        
    with open(out_dir / "metadata.json", "w") as f:
        json.dump({target_seq: saved_meta}, f, indent=2)
        
    log.info(f"Done! Saved metadata to {out_dir / 'metadata.json'}")

if __name__ == "__main__":
    main()
