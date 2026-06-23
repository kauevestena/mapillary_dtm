import sys
from pathlib import Path
sys.path.insert(0, ".")
import logging
logging.basicConfig(level=logging.INFO)

from tests.sample_loader import get_sample_frames
seqs, _ = get_sample_frames()

import json
import numpy as np
from pathlib import Path
dataset_dir = Path("qa/bbox_test/cache/opensfm/c6ixrty4mug2hb0gebz9rm")
with open(dataset_dir / "reconstruction.json") as f:
    data = json.load(f)

from dtm_from_mapillary.common_core import ReconstructionResult
points_array = np.zeros((0, 3))
recon_pts = data[0].get("points", {})
points_xyz = []
for p in recon_pts.values():
    points_xyz.append(p["coordinates"])
points_array = np.array(points_xyz) if points_xyz else np.zeros((0, 3))

pose_map = {}
for cam_id_with_ext, shot in data[0].get("shots", {}).items():
    cam_id = cam_id_with_ext.split('.')[0]
    if "translation" in shot:
        import cv2
        R, _ = cv2.Rodrigues(np.array(shot["rotation"]))
        t = np.array(shot["translation"])
        from dtm_from_mapillary.common_core import Pose
        t_w = -R.T @ t
        pose_map[cam_id] = Pose(R=R.T, t=t_w)

recon = ReconstructionResult(seq_id="c6ixrty4mug2hb0gebz9rm", source="opensfm", poses=pose_map, frames=None, points_xyz=points_array)

with open("qa/bbox_test/metadata/camera_positions_opensfm.geojson") as f:
    gj = json.load(f)
frames_list = []
from dtm_from_mapillary.common_core import FrameMeta
for feat in gj["features"]:
    props = feat["properties"]
    if props["seq_id"] == "c6ixrty4mug2hb0gebz9rm":
        frames_list.append(FrameMeta(
            image_id=props["image_id"],
            seq_id=props["seq_id"],
            captured_at_ms=props["captured_at_ms"],
            lon=feat["geometry"]["coordinates"][0],
            lat=feat["geometry"]["coordinates"][1],
            alt_ellip=feat["geometry"]["coordinates"][2] if len(feat["geometry"]["coordinates"]) > 2 else 0.0,
            camera_type="unknown",
            cam_params={},
            quality_score=0.0,
            thumbnail_url=""
        ))
reconA = {"c6ixrty4mug2hb0gebz9rm": recon}

from dtm_from_mapillary.geom.height_solver import solve_scale_and_h
print("frames_list[0].alt_ellip:", frames_list[0].alt_ellip)
gnss_frames = [f for f in frames_list if f.alt_ellip is not None]
print("gnss_frames len:", len(gnss_frames))
print("recon:", recon is not None)
print("recon.poses:", len(recon.poses))
anchors, scales, heights = solve_scale_and_h(reconA, {}, {}, [], {"c6ixrty4mug2hb0gebz9rm": frames_list}, mask_dir=Path("qa/bbox_test/cache/masks"))
print("Scales:", scales)
print("Heights:", heights)
