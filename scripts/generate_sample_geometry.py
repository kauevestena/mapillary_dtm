#!/usr/bin/env python3
import sys
import os
import json
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from tests.sample_loader import get_sample_frames

# Import the actual core pipeline modules! No mocks!
from dtm_from_mapillary.geom import sfm_opensfm, sfm_colmap, vo_simplified, sfm_dim

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("generate_geometry")

# Strictly enforce Docker images for the real binary executions
os.environ["COLMAP_DOCKER_IMAGE"] = "colmap/colmap:latest"
os.environ["OPEN_SFM_DOCKER_IMAGE"] = "freakthemighty/opensfm:latest"

# Strictly disable synthetic fallbacks

def np_encoder(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def main():
    log.info("Loading QA sample dataset...")
    seqs, imagery_root = get_sample_frames()
    
    geom_dir = ROOT / "qa" / "data" / "sample_dataset" / "geometry"
    geom_dir.mkdir(parents=True, exist_ok=True)
    
    # ------------------------------------------------------------------
    # TRACK A: OpenSfM
    # ------------------------------------------------------------------
    opensfm_ws = geom_dir / "opensfm"
    log.info(f"Running Track A (OpenSfM) in {opensfm_ws}")
    try:
        opensfm_results = sfm_opensfm.run(
            seqs,
            imagery_root=imagery_root,
            workspace_root=opensfm_ws,
            force=True, # force re-run to guarantee we are testing it now
            progress=True
        )
        log.info(f"OpenSfM completed. Outputs serialized in {opensfm_ws}")
    except Exception as e:
        log.error(f"OpenSfM Track Failed: {e}")
        sys.exit(1)
        
    # ------------------------------------------------------------------
    # TRACK B: COLMAP
    # ------------------------------------------------------------------
    colmap_ws = geom_dir / "colmap"
    log.info(f"Running Track B (COLMAP) in {colmap_ws}")
    try:
        colmap_results = sfm_dim.run(
            seqs,
            imagery_root=imagery_root,
            workspace_root=colmap_ws,
            progress=True
        )
        log.info(f"COLMAP completed. Outputs serialized in {colmap_ws}")
    except Exception as e:
        log.error(f"COLMAP Track Failed: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # TRACK C: Visual Odometry (OpenCV)
    # ------------------------------------------------------------------
    vo_ws = geom_dir / "vo"
    vo_ws.mkdir(parents=True, exist_ok=True)
    log.info(f"Running Track C (Visual Odometry) saving to {vo_ws}")
    try:
        vo_results = vo_simplified.run(
            seqs,
            imagery_root=imagery_root,
            progress=True
        )
        
        # Serialize VO results manually since it's pure Python
        serialized_vo = {}
        for seq_id, result in vo_results.items():
            poses = {}
            for img_id, pose in result.poses.items():
                poses[img_id] = {
                    "R": pose.R.tolist(),
                    "t": pose.t.tolist()
                }
            serialized_vo[seq_id] = {
                "poses": poses,
                "points_xyz": result.points_xyz.tolist(),
                "metadata": result.metadata
            }
            
        with open(vo_ws / "vo_results.json", "w") as f:
            json.dump(serialized_vo, f, indent=2, default=np_encoder)
        
        log.info(f"VO completed. Outputs serialized in {vo_ws}")
    except Exception as e:
        log.error(f"VO Track Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
