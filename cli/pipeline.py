"""
Typer CLI orchestrator for the DTM-from-Mapillary pipeline.
"""
from __future__ import annotations
import os, json, typer
from typing import Optional
from ..ingest.sequence_scan import discover_sequences
from ..ingest.sequence_filter import filter_car_sequences
from ..semantics.ground_masks import prepare as prepare_masks
from ..geom.sfm_opensfm import run as run_opensfm
from ..geom.sfm_colmap import run as run_colmap
from ..geom.vo_simplified import run as run_vo
from ..geom.anchors import find_anchors
from ..geom.height_solver import solve_scale_and_h
from ..ground.ground_extract_3d import label_and_filter_points
from ..ground.recon_consensus import agree as consensus_agree
from ..fusion.heightmap_fusion import fuse as fuse_heightmap
from ..fusion.smoothing_regularization import edge_aware
from ..qa.qa_internal import slope_from_plane_fit
from ..io.writers import write_geotiffs, write_laz
from ..qa.reports import write_html

app = typer.Typer(help="DTM from Mapillary — high-accuracy pipeline")

@app.command()
def run(aoi_bbox: str,
        out_dir: str = "./out",
        token: Optional[str] = None):
    """
    Run the full pipeline over an AOI bbox: "lon_min,lat_min,lon_max,lat_max".
    """
    os.makedirs(out_dir, exist_ok=True)
    bbox = tuple(map(float, aoi_bbox.split(",")))
    seqs = discover_sequences(bbox, token=token)
    seqs = filter_car_sequences(seqs)
    prepare_masks(seqs)

    reconA = run_opensfm(seqs)
    reconB = run_colmap(seqs)
    vo = run_vo(seqs)

    anchors = find_anchors(seqs, token=token)
    scales, heights = solve_scale_and_h(reconA, reconB, vo, anchors, seqs)

    ptsA = label_and_filter_points(reconA, scales)
    ptsB = label_and_filter_points(reconB, scales)
    # Placeholder for VO+mono-derived ground points
    ptsC = []

    pts = consensus_agree(ptsA, ptsB, ptsC)
    dtm, conf = fuse_heightmap(pts)
    dtm_s = edge_aware(dtm)
    slope_deg, aspect = slope_from_plane_fit(dtm_s)

    # Writers (transforms/CRS omitted in scaffold)
    write_geotiffs(out_dir, dtm_s, slope_deg, conf, transform=None, crs="EPSG:4979")
    # Write LAZ (attrs omitted in scaffold)
    import numpy as np
    write_laz(out_dir, np.zeros((0,3), dtype=np.float32))

    manifest = {
        "bbox": bbox,
        "scales": {k: float(v) for k,v in (scales or {}).items()},
        "heights": {k: float(v) for k,v in (heights or {}).items()},
    }
    write_html(out_dir, manifest)

if __name__ == "__main__":
    app()
