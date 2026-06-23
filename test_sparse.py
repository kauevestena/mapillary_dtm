import sys
from pathlib import Path
sys.path.insert(0, ".")
from dtm_from_mapillary.geom.opensfm_adapter import _load_opensfm
from tests.sample_loader import get_sample_frames
import numpy as np

seqs, _ = get_sample_frames()
dataset_dir = Path("qa/bbox_test/reconstructions/opensfm/sj6mkthsm53cmb1kxqfr46")
if not dataset_dir.exists():
    print(f"Dataset {dataset_dir} not found")
    sys.exit(0)

recon = _load_opensfm(dataset_dir)
if recon is None or recon.points_xyz is None:
    print("No reconstruction or points")
    sys.exit(0)

pts = recon.points_xyz
print(f"Total points: {len(pts)}")

# For simplicity, we just check if any points are below the cameras
cam_heights = [p.t[2] for p in recon.poses.values()]
if not cam_heights:
    sys.exit(0)
med_cam_z = np.median(cam_heights)

below = [p for p in pts if p[2] < med_cam_z - 0.5]
print(f"Points >0.5m below cameras: {len(below)}")
