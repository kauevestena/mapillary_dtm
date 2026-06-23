import sys
from pathlib import Path
sys.path.insert(0, ".")
from tests.sample_loader import get_sample_frames
seqs, _ = get_sample_frames()

import json
import numpy as np

dataset_dir = Path("qa/bbox_test/cache/opensfm/c6ixrty4mug2hb0gebz9rm")
recon_path = dataset_dir / "reconstruction.json"
with open(recon_path) as f:
    data = json.load(f)

sfm_positions = []
gnss_positions = []

with open("qa/bbox_test/metadata/camera_positions_opensfm.geojson") as f:
    gj = json.load(f)
frames = []
from dtm_from_mapillary.common_core import FrameMeta
for feat in gj["features"]:
    props = feat["properties"]
    if props["seq_id"] == "c6ixrty4mug2hb0gebz9rm":
        frames.append(FrameMeta(
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

result = ReconstructionResult(seq_id="c6ixrty4mug2hb0gebz9rm", source="opensfm", poses=pose_map, frames=frames, points_xyz=points_array)

from dtm_from_mapillary.geom.utils import umeyama_alignment
sfm_positions = np.array(sfm_positions)
gnss_positions = np.array(gnss_positions)
R, t, s = umeyama_alignment(sfm_positions, gnss_positions)

print("Original R:")
print(R)

if R[2, 2] < 0:
    print("Z is flipped! Fixing...")
    src_mean = sfm_positions.mean(axis=0)
    dst_mean = gnss_positions.mean(axis=0)
    src_demean = sfm_positions - src_mean
    dst_demean = gnss_positions - dst_mean
    A = dst_demean.T @ src_demean / sfm_positions.shape[0]
    U, S, Vt = np.linalg.svd(A)
    U[:, 1] = -U[:, 1]
    U[:, 2] = -U[:, 2]
    R_fixed = U @ Vt
    print("Fixed R:")
    print(R_fixed)
    
    from scipy.spatial.transform import Rotation
    print("Fixed Euler:", Rotation.from_matrix(R_fixed).as_euler("xyz", degrees=True))


