#!/usr/bin/env python3
import sys
import logging
import csv
import math
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.sample_loader import get_sample_frames
from dtm_from_mapillary.geom import sfm_opensfm, sfm_dim, vo_simplified
from dtm_from_mapillary.geom.anchors import find_anchors
from dtm_from_mapillary.geom.height_solver import solve_scale_and_h, estimate_h_cam_from_dtm
from dtm_from_mapillary.ground.ground_extract_3d import label_and_filter_points
from dtm_from_mapillary.fusion.heightmap_fusion import fuse
from dtm_from_mapillary.cli.pipeline import _infer_origin
from dtm_from_mapillary.common_core import enu_to_wgs84

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("generate_sample_ground")

def main():
    log.info("Loading QA sample dataset...")
    seqs, imagery_root = get_sample_frames()
    
    geom_dir = ROOT / "qa" / "data" / "sample_dataset" / "geometry"
    mask_dir = ROOT / "qa" / "data" / "sample_dataset" / "masks"
    depth_dir = ROOT / "qa" / "data" / "sample_dataset" / "depth"
    
    # Load geometry tracks
    log.info("Loading OpenSfM results...")
    opensfm_results = sfm_opensfm.run(seqs, imagery_root=imagery_root, workspace_root=geom_dir / "opensfm", force=False)
    
    log.info("Loading COLMAP results...")
    colmap_results = sfm_dim.run(seqs, imagery_root=imagery_root, workspace_root=geom_dir / "colmap")
    
    log.info("Loading VO results...")
    vo_results = vo_simplified.run(seqs, imagery_root=imagery_root)
    
    log.info("Solving scale & height (pass 1: Umeyama alignment + default h_cam)...")
    anchors = find_anchors(seqs)
    scales, heights = solve_scale_and_h(opensfm_results, colmap_results, vo_results, anchors, seqs)
    
    reconstructions = {"opensfm": opensfm_results, "colmap": colmap_results, "vo": vo_results}
    
    # ── Pass 1: extract ground points with default h_cam, build DTM ──
    log.info("Pass 1: Extracting ground points with initial h_cam...")
    extracted = {}
    for name, recon in reconstructions.items():
        include_mono = (name != "vo")
        extracted[name] = label_and_filter_points(
            recon,
            scales,
            mask_dir=mask_dir,
            mono_cache=depth_dir,
            vo_recon=vo_results,
            imagery_root=imagery_root,
            include_plane_sweep=False,
            include_monodepth=include_mono,
            include_sparse=(name != "vo"),
            heights=heights,
        )
    
    all_points = []
    for pts in extracted.values():
        all_points.extend(pts)
        
    log.info(f"Fused {len(all_points)} total ground points. Building DTM...")
    points_dicts = [{"x": p.x, "y": p.y, "z": p.z, "sem_prob": p.sem_prob, "uncertainty": p.uncertainty_m} for p in all_points]
    dtm, conf, grid = fuse(points_dicts, grid_res=0.5, return_grid=True)
    
    # ── Pass 2: estimate h_cam from DTM for each reconstruction ──
    log.info("Pass 2: Estimating h_cam from DTM under camera positions...")
    lon0, lat0, h0 = _infer_origin(seqs, None)
    
    # Flatten frames map
    frames_map = {}
    for seq_frames in seqs.values():
        for f in seq_frames:
            frames_map[f.image_id] = f

    for name in ["opensfm", "colmap", "vo"]:
        res_map = reconstructions[name]
        out_csv = ROOT / "qa" / "data" / "sample_dataset" / f"height_evaluation_{name}.csv"
        
        # Estimate h_cam per sequence using actual DTM values
        h_cam_per_seq = estimate_h_cam_from_dtm(res_map, scales, dtm, grid)
        
        csv_data = []
        for seq_id, result in res_map.items():
            if not result.poses:
                continue
                
            scale = scales.get(seq_id, 1.0)
            h_cam = h_cam_per_seq.get(seq_id, 2.0)
            
            for img_id, pose in result.poses.items():
                frame = frames_map.get(img_id)
                if not frame:
                    continue
                    
                camera_x, camera_y, camera_z = pose.t * scale
                _, _, computed_trajectory_height = enu_to_wgs84(camera_x, camera_y, camera_z, lon0, lat0, h0)
                
                # Ground height = trajectory height minus the LS-fitted constant h_cam
                computed_height = computed_trajectory_height - h_cam
                computed_height_diff = h_cam
                    
                csv_data.append({
                    "image_id": img_id,
                    "metadata_height": frame.alt_ellip if frame.alt_ellip is not None else "",
                    "computed_trajectory_height": round(computed_trajectory_height, 3),
                    "computed_height_difference": round(computed_height_diff, 3),
                    "computed_height": round(computed_height, 3),
                })
                
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_id", "metadata_height", "computed_trajectory_height", "computed_height_difference", "computed_height"])
            writer.writeheader()
            for row in sorted(csv_data, key=lambda x: x["image_id"]):
                writer.writerow(row)
        
        # Report per-sequence h_cam values
        for seq_id, h in h_cam_per_seq.items():
            log.info(f"  {name} / {seq_id}: h_cam = {h:.3f} m")
        log.info(f"Wrote table for {name} with {len(csv_data)} frames to {out_csv}")

if __name__ == "__main__":
    main()
