#!/usr/bin/env python3
import sys
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.sample_loader import get_sample_frames
from dtm_from_mapillary.geom import sfm_opensfm, sfm_colmap, vo_simplified, sfm_dim
from dtm_from_mapillary.geom.anchors import find_anchors
from dtm_from_mapillary.geom.height_solver import solve_scale_and_h
from dtm_from_mapillary.io.geojson_writers import write_all_camera_positions_geojson
from dtm_from_mapillary.cli.pipeline import _infer_origin

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("generate_geojson")

def main():
    log.info("Loading QA sample dataset...")
    seqs, imagery_root = get_sample_frames()
    
    geom_dir = ROOT / "qa" / "data" / "sample_dataset" / "geometry"
    geojson_dir = ROOT / "qa" / "data" / "sample_dataset" / "geojson"
    geojson_dir.mkdir(parents=True, exist_ok=True)
    
    log.info("Loading existing OpenSfM results...")
    opensfm_results = sfm_opensfm.run(
        seqs,
        imagery_root=imagery_root,
        workspace_root=geom_dir / "opensfm",
        force=False,
    )
    
    log.info("Loading existing COLMAP results...")
    colmap_results = sfm_dim.run(
        seqs,
        imagery_root=imagery_root,
        workspace_root=geom_dir / "colmap",
    )
    
    log.info("Loading VO results...")
    vo_results = vo_simplified.run(
        seqs,
        imagery_root=imagery_root,
    )
    
    log.info("Finding anchors...")
    # Passing no token, it uses mock if needed or just empty
    anchors = find_anchors(seqs)
    
    log.info("Solving scale and height...")
    scales, heights = solve_scale_and_h(opensfm_results, colmap_results, vo_results, anchors, seqs)
    
    # We need a bbox to infer the origin. We can just use dummy bbox since _infer_origin uses sequences
    # wait, _infer_origin takes bbox: tuple[float, float, float, float] | None = None
    lon0, lat0, h0 = _infer_origin(seqs, None)
    
    reconstructions = {
        "opensfm": opensfm_results,
        "colmap": colmap_results,
        "vo": vo_results,
    }
    
    log.info(f"Writing calibrated GeoJSON files to {geojson_dir}...")
    paths = write_all_camera_positions_geojson(reconstructions, lon0, lat0, h0, scales, geojson_dir)
    
    for path in paths:
        log.info(f"Generated GeoJSON: {path}")

if __name__ == "__main__":
    main()
