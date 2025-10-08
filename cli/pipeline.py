"""
Typer CLI orchestrator for the DTM-from-Mapillary pipeline.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

try:  # Optional dependency for CLI usage
    import typer
except ImportError:  # pragma: no cover - typer is optional for library usage
    typer = None  # type: ignore[assignment]

from .. import constants
from ..geom.anchors import find_anchors
from ..geom.height_solver import solve_scale_and_h
from ..geom.sfm_colmap import run as run_colmap
from ..geom.sfm_opensfm import run as run_opensfm
from ..geom.vo_simplified import run as run_vo
from ..ground.corridor_fill_tin import (
    build_tin,
    build_constrained_tin,
    corridor_to_local,
    sample_outside_corridor,
)
from ..ground.ground_extract_3d import label_and_filter_points
from ..ground.recon_consensus import agree as consensus_agree
from ..ground.breakline_integration import (
    project_curbs_to_3d,
    merge_breakline_segments,
    simplify_breaklines,
    densify_breaklines,
)
from ..ingest.sequence_filter import filter_car_sequences
from ..ingest.sequence_scan import discover_sequences
from ..io.writers import write_geotiffs, write_laz
from ..osm.osmnx_utils import corridor_from_osm_bbox
from ..qa.qa_external import compare_to_geotiff
from ..qa.qa_internal import slope_from_plane_fit, write_agreement_maps
from ..qa.reports import write_html
from ..semantics.curb_edge_lane import extract_curbs_and_lanes
from ..semantics.ground_masks import prepare as prepare_masks
from ..fusion.heightmap_fusion import fuse as fuse_heightmap
from ..fusion.smoothing_regularization import edge_aware

log = logging.getLogger(__name__)


def run_pipeline(
    aoi_bbox: str,
    out_dir: str = "./out",
    token: Optional[str] = None,
    use_learned_uncertainty: bool = False,
    uncertainty_model_path: Optional[str] = None,
    enforce_breaklines: bool = False,
) -> dict:
    """
    Run the full pipeline over an AOI bbox: "lon_min,lat_min,lon_max,lat_max".
    Returns the manifest describing the run.

    Parameters
    ----------
    aoi_bbox : str
        Bounding box as "lon_min,lat_min,lon_max,lat_max"
    out_dir : str
        Output directory for results
    token : str, optional
        Mapillary API token (or use env var / file)
    use_learned_uncertainty : bool
        Enable learned uncertainty calibration (default: False)
    uncertainty_model_path : str, optional
        Path to saved uncertainty model (will train if not found)
    enforce_breaklines : bool
        Enable breakline enforcement in TIN (default: False)
    """
    os.makedirs(out_dir, exist_ok=True)
    bbox = tuple(map(float, aoi_bbox.split(",")))
    seqs = discover_sequences(bbox, token=token)
    seqs = filter_car_sequences(seqs)
    mask_index = prepare_masks(seqs)
    curb_lines = extract_curbs_and_lanes(seqs)

    reconA = run_opensfm(seqs)
    reconB = run_colmap(seqs)
    vo = run_vo(seqs)

    anchors = find_anchors(seqs, token=token)
    scales, heights = solve_scale_and_h(reconA, reconB, vo, anchors, seqs)

    ptsA = label_and_filter_points(reconA, scales)
    ptsB = label_and_filter_points(reconB, scales)
    # Placeholder for VO+mono-derived ground points
    ptsC: list = []

    # Apply learned uncertainty calibration if requested
    if use_learned_uncertainty:
        try:
            from ..ml.integration import (
                load_or_train_calibrator,
                apply_learned_uncertainty,
            )

            model_path = Path(
                uncertainty_model_path or (Path(out_dir) / "uncertainty_model.pkl")
            )

            # For training, we need consensus first (will use for next iteration)
            # For now, apply to individual tracks
            log.info("Learned uncertainty calibration enabled")

            # This is a simplified approach - full implementation would train on previous runs
            # Here we just demonstrate the integration point
            if model_path.exists():
                calibrator = load_or_train_calibrator(model_path)
                ptsA = apply_learned_uncertainty(ptsA, calibrator)
                ptsB = apply_learned_uncertainty(ptsB, calibrator)
                log.info("Applied learned uncertainty to tracks A and B")
        except ImportError as exc:
            log.warning(
                "ML dependencies not available for learned uncertainty: %s", exc
            )
        except Exception as exc:
            log.warning("Failed to apply learned uncertainty: %s", exc)

    consensus = consensus_agree(ptsA, ptsB, ptsC)

    # Breakline processing (if enabled)
    breakline_stats = {
        "enabled": enforce_breaklines,
        "curbs_detected": len(curb_lines) if curb_lines else 0,
        "breaklines_3d": 0,
        "vertices": 0,
        "edges": 0,
    }
    breakline_vertices = None
    breakline_edges = None

    if enforce_breaklines and curb_lines:
        try:
            log.info("Breakline enforcement enabled - processing curbs")

            # Get camera poses and models from Track A (primary reconstruction)
            camera_poses = {}
            camera_models = {}
            if reconA and "cameras" in reconA and "poses" in reconA:
                camera_poses = reconA["poses"]
                camera_models = reconA["cameras"]

            # Project curbs to 3D
            breaklines_3d = project_curbs_to_3d(
                curbs=curb_lines,
                camera_poses=camera_poses,
                camera_models=camera_models,
                consensus_points=consensus,
            )
            breakline_stats["breaklines_3d"] = len(breaklines_3d)
            log.info("Projected %d curb detections to 3D", len(breaklines_3d))

            if breaklines_3d:
                # Merge overlapping segments
                merged = merge_breakline_segments(breaklines_3d)
                log.info("Merged into %d polylines", len(merged))

                # Simplify
                simplified = simplify_breaklines(merged)
                log.info("Simplified to %d polylines", len(simplified))

                # Densify and get constraints
                breakline_vertices, breakline_edges = densify_breaklines(simplified)
                breakline_stats["vertices"] = len(breakline_vertices)
                breakline_stats["edges"] = len(breakline_edges)
                log.info(
                    "Densified breaklines: %d vertices, %d edges",
                    len(breakline_vertices),
                    len(breakline_edges),
                )

        except Exception as exc:
            log.warning("Breakline processing failed: %s", exc)
            enforce_breaklines = False  # Disable for TIN construction

    corridor_info = None
    tin_samples: list = []
    try:
        corridor_raw = corridor_from_osm_bbox(bbox)
        lon0, lat0, h0 = _infer_origin(seqs, bbox)
        corridor_info = corridor_to_local(corridor_raw, lon0=lon0, lat0=lat0, h0=h0)
        if corridor_info:
            # Build TIN with optional breakline constraints
            if enforce_breaklines and breakline_vertices is not None and breakline_edges:
                log.info("Building constrained TIN with breakline enforcement")
                tin_model = build_constrained_tin(
                    points=consensus,
                    breakline_vertices=breakline_vertices,
                    breakline_edges=breakline_edges,
                )
            else:
                tin_model = build_tin(consensus)

            tin_samples = sample_outside_corridor(
                consensus,
                corridor_info,
                grid_res=constants.GRID_RES_M,
                max_extrapolation_m=constants.MAX_TIN_EXTRAPOLATION_M,
                tin=tin_model,
            )
    except Exception as exc:  # pragma: no cover - corridor is best-effort
        log.warning("Corridor/TIN expansion failed: %s", exc)

    consensus_all = list(consensus)
    if tin_samples:
        consensus_all.extend(tin_samples)

    dtm, conf = fuse_heightmap(consensus_all)
    dtm_s = edge_aware(dtm)
    slope_deg, aspect = slope_from_plane_fit(dtm_s)

    # Writers (transforms/CRS omitted in scaffold)
    geotiff_paths = write_geotiffs(
        out_dir, dtm_s, slope_deg, conf, transform=None, crs="EPSG:4979"
    )
    laz_path = write_laz(out_dir, np.zeros((0, 3), dtype=np.float32))

    qa_dir = Path(out_dir) / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    agreement_path = qa_dir / "agreement_maps.npz"
    agreement_results = write_agreement_maps(agreement_path, dtm_s, {})
    agreement_summary = _summarize_agreement(agreement_results)

    external_stats = None
    dtm_key = "dtm_0p5m_ellipsoid.tif"
    qa_reference = Path("qa/data/qa_dtm_4326.tif")
    if dtm_key in geotiff_paths and qa_reference.exists():
        try:
            external_stats = compare_to_geotiff(
                geotiff_paths[dtm_key], str(qa_reference)
            )
        except Exception as exc:  # pragma: no cover
            log.warning("External QA comparison failed: %s", exc)

    manifest = {
        "bbox": bbox,
        "scales": {k: float(v) for k, v in (scales or {}).items()},
        "heights": {k: float(v) for k, v in (heights or {}).items()},
        "corridor_source": corridor_info.get("source") if corridor_info else None,
        "corridor_buffer_m": corridor_info.get("buffer_m") if corridor_info else None,
        "tin_samples": len(tin_samples),
        "breaklines": breakline_stats,
        "outputs": {
            "geotiffs": geotiff_paths,
            "laz": laz_path,
            "agreement_maps": str(agreement_path),
        },
        "qa": {
            "agreement_summary": agreement_summary,
            "external": external_stats,
        },
        "constants": _constants_snapshot(),
    }
    qa_metrics = {}
    qa_metrics.update({f"agreement_{k}": v for k, v in agreement_summary.items()})
    if external_stats:
        qa_metrics.update({f"external_{k}": v for k, v in external_stats.items()})

    artifact_paths = {
        "DTM": geotiff_paths.get(dtm_key, ""),
        "Slope": geotiff_paths.get("slope_deg.tif", ""),
        "Confidence": geotiff_paths.get("confidence.tif", ""),
        "LAZ / NPZ": laz_path,
        "Agreement maps": str(agreement_path),
    }
    write_html(out_dir, manifest, qa_summary=qa_metrics, artifact_paths=artifact_paths)
    return manifest


# Backwards-compatible alias for library callers.
run = run_pipeline

if typer is not None:
    app = typer.Typer(help="DTM from Mapillary — high-accuracy pipeline")

    @app.command("run")
    def cli_run(
        aoi_bbox: str,
        out_dir: str = "./out",
        token: Optional[str] = None,
        use_learned_uncertainty: bool = False,
        uncertainty_model_path: Optional[str] = None,
        enforce_breaklines: bool = False,
    ) -> None:
        """Run the DTM generation pipeline.

        Parameters
        ----------
        aoi_bbox : str
            Area of interest as "lon_min,lat_min,lon_max,lat_max"
        out_dir : str
            Output directory (default: ./out)
        token : str, optional
            Mapillary API token
        use_learned_uncertainty : bool
            Enable ML-based uncertainty calibration
        uncertainty_model_path : str, optional
            Path to uncertainty model file
        enforce_breaklines : bool
            Enable breakline enforcement in TIN construction
        """
        run_pipeline(
            aoi_bbox=aoi_bbox,
            out_dir=out_dir,
            token=token,
            use_learned_uncertainty=use_learned_uncertainty,
            uncertainty_model_path=uncertainty_model_path,
            enforce_breaklines=enforce_breaklines,
        )

else:  # pragma: no cover - only executed when typer missing
    app = None


def _infer_origin(seqs, bbox):
    for frames in seqs.values():
        if frames:
            frame = frames[0]
            h = frame.alt_ellip or 0.0
            return float(frame.lon), float(frame.lat), float(h)
    lon_min, lat_min, lon_max, lat_max = bbox
    lon0 = (lon_min + lon_max) * 0.5
    lat0 = (lat_min + lat_max) * 0.5
    return float(lon0), float(lat0), 0.0


def _summarize_agreement(results: dict) -> dict:
    summary = {}
    for key, arr in results.items():
        if not isinstance(arr, np.ndarray):
            summary[key] = arr
            continue
        if arr.size == 0 or np.isnan(arr).all():
            summary[key] = {"mean": None, "p95": None}
            continue
        mean = float(np.nanmean(arr))
        p95 = (
            float(np.nanpercentile(arr, 95)) if np.isfinite(arr).any() else float("nan")
        )
        if not np.isfinite(mean):
            mean = None
        if not np.isfinite(p95):
            p95 = None
        summary[key] = {"mean": mean, "p95": p95}
    return summary


def _constants_snapshot() -> dict:
    snapshot = {}
    for name in dir(constants):
        if name.isupper():
            snapshot[name] = getattr(constants, name)
    return snapshot


if __name__ == "__main__":
    if typer is None:
        raise RuntimeError(
            "typer is required to run the CLI. Install `typer` to use this entry point."
        )
    app()
