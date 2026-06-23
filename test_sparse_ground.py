import sys
from pathlib import Path
sys.path.insert(0, ".")
from tests.sample_loader import get_sample_frames
seqs, _ = get_sample_frames()

dataset_dir = Path("qa/bbox_test/cache/opensfm/c6ixrty4mug2hb0gebz9rm")
recon = dataset_dir / "reconstruction.json"
if recon is None:
    print("Recon not found")
    sys.exit()

import json
import numpy as np
from dtm_from_mapillary.common_core import ReconstructionResult, Pose

with open(recon) as f:
    data = json.load(f)

pose_map = {}
for cam_id, shot in data[0].get("shots", {}).items():
    if "translation" in shot:
        import cv2
        R, _ = cv2.Rodrigues(np.array(shot["rotation"]))
        t = np.array(shot["translation"])
        t_w = -R.T @ t
        pose_map[cam_id] = Pose(R=R.T, t=t_w)

recon_pts = data[0].get("points", {})
points_xyz = []
for p in recon_pts.values():
    points_xyz.append(p["coordinates"])
points_array = np.array(points_xyz) if points_xyz else np.zeros((0, 3))

from dtm_from_mapillary.common_core import FrameMeta
dummy_frames = []
for cam_id in pose_map.keys():
    dummy_frames.append(FrameMeta(image_id=cam_id, seq_id="c6ixrty4mug2hb0gebz9rm", captured_at_ms=0, lon=0.0, lat=0.0, alt_ellip=None, camera_type="unknown", cam_params={}, quality_score=None, thumbnail_url=None))

result = ReconstructionResult(seq_id="c6ixrty4mug2hb0gebz9rm", source="opensfm", poses=pose_map, frames=dummy_frames, points_xyz=points_array)

from dtm_from_mapillary.ground.ground_extract_3d import label_and_filter_points
ground_points = label_and_filter_points(
    {"c6ixrty4mug2hb0gebz9rm": result}, 
    {"c6ixrty4mug2hb0gebz9rm": 1.0}, 
    mask_dir="qa/bbox_test/cache/masks",
    include_sparse=True, 
    include_monodepth=False, 
    include_plane_sweep=False
)
from dtm_from_mapillary.geom.height_solver import _estimate_h_cam_ls
try:
    h_cam = _estimate_h_cam_ls(result, result.frames, [], ground_points)
    print(f"Estimated h_cam: {h_cam}")
except ValueError as e:
    print(f"ValueError: {e}")
