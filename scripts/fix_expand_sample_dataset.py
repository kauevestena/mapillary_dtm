#!/usr/bin/env python3
"""
Fix and expand the QA sample dataset:
1. Re-download proper metadata (camera_parameters, altitude, width, height, etc.)
   for the existing 4 images (which had zeroed-out camera_parameters).
2. Expand the dataset by downloading neighboring images from the same Mapillary sequence.
"""

import sys
import os
import json
import logging
import requests
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("fix_expand_dataset")

DATASET_DIR = ROOT / "qa" / "data" / "sample_dataset"
IMAGERY_DIR = DATASET_DIR / "imagery"
METADATA_PATH = DATASET_DIR / "metadata.json"

FIELDS = "id,camera_parameters,camera_type,width,height,make,model,altitude,captured_at,geometry,thumb_original_url,sequence,quality_score,computed_rotation,computed_geometry"

# Number of extra images to download on each side of our existing 4
EXPAND_BEFORE = 4
EXPAND_AFTER = 4


def get_token() -> str:
    token_path = ROOT / "mapillary_token"
    if token_path.exists():
        return token_path.read_text().strip()
    env = os.getenv("MAPILLARY_TOKEN")
    if env:
        return env.strip()
    raise RuntimeError("No Mapillary token found")


def fetch_image_metadata(image_id: str, token: str) -> dict:
    url = f"https://graph.mapillary.com/{image_id}"
    params = {"access_token": token, "fields": FIELDS}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def download_thumbnail(url: str, dest: Path) -> bool:
    if dest.exists():
        log.info(f"  Already cached: {dest.name}")
        return True
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        log.info(f"  Downloaded: {dest.name} ({len(r.content)} bytes)")
        return True
    except Exception as e:
        log.error(f"  Failed to download {dest.name}: {e}")
        return False


def build_frame_entry(meta: dict, seq_id: str) -> dict:
    geom = meta.get("geometry") or meta.get("computed_geometry") or {}
    coords = geom.get("coordinates", [0, 0])

    cam_params = meta.get("camera_parameters", [])
    if not cam_params or not isinstance(cam_params, list):
        cam_params = [0.0, 0.0, 0.0]

    return {
        "image_id": str(meta["id"]),
        "seq_id": seq_id,
        "captured_at_ms": meta.get("captured_at", 0),
        "lon": coords[0],
        "lat": coords[1],
        "altitude": meta.get("altitude"),
        "camera_type": meta.get("camera_type", "unknown"),
        "camera_parameters": cam_params,
        "width": meta.get("width"),
        "height": meta.get("height"),
        "make": meta.get("make"),
        "model": meta.get("model"),
        "computed_rotation": meta.get("computed_rotation"),
        "quality_score": meta.get("quality_score", 0.5),
        "thumbnail_url": meta.get("thumb_original_url"),
    }


def main():
    token = get_token()
    seq_id = "l27kwlcx3fjh7t6w9ccvic"

    # 1. Get all image IDs in the sequence
    log.info("Fetching full sequence image list...")
    url = "https://graph.mapillary.com/image_ids"
    params = {"access_token": token, "sequence_id": seq_id}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    all_ids = [d["id"] for d in r.json().get("data", [])]
    log.info(f"Sequence has {len(all_ids)} images total")

    # 2. Find our existing 4 images
    our_ids = {"549798756007659", "177246630949005", "170058561705143", "530796564614344"}
    our_positions = [i for i, img_id in enumerate(all_ids) if img_id in our_ids]
    log.info(f"Our 4 images are at positions: {our_positions}")

    # 3. Determine expansion range
    min_pos = min(our_positions)
    max_pos = max(our_positions)
    start = max(0, min_pos - EXPAND_BEFORE)
    end = min(len(all_ids) - 1, max_pos + EXPAND_AFTER)
    target_ids = all_ids[start:end + 1]
    log.info(f"Target range: positions {start}-{end} ({len(target_ids)} images)")

    # 4. Fetch metadata and download imagery for each target image
    frames = []
    seq_imagery_dir = IMAGERY_DIR / seq_id

    for img_id in target_ids:
        log.info(f"Fetching metadata for {img_id}...")
        meta = fetch_image_metadata(img_id, token)

        entry = build_frame_entry(meta, seq_id)
        frames.append(entry)

        # Download thumbnail
        thumb_url = meta.get("thumb_original_url")
        if thumb_url:
            # Use the resolution-tagged naming: {image_id}_1024.jpg
            dest = seq_imagery_dir / f"{img_id}_1024.jpg"
            download_thumbnail(thumb_url, dest)

    # 5. Sort by capture time
    frames.sort(key=lambda f: f["captured_at_ms"])

    # 6. Add local_path references
    for frame in frames:
        img_id = frame["image_id"]
        rel_path = f"qa/data/sample_dataset/imagery/{seq_id}/{img_id}_1024.jpg"
        frame["local_path"] = rel_path

    # 7. Write updated metadata.json
    metadata = {seq_id: frames}
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log.info(f"Wrote {len(frames)} frames to {METADATA_PATH}")

    # 8. Summary
    for f in frames:
        cam = f["camera_parameters"]
        marker = " <-- ORIGINAL" if f["image_id"] in our_ids else " (NEW)"
        log.info(
            f"  {f['image_id']} | focal={cam[0]:.4f} k1={cam[1]:.4f} k2={cam[2]:.4f} | "
            f"alt={f.get('altitude', '?')} | {f['width']}x{f['height']}{marker}"
        )


if __name__ == "__main__":
    main()
