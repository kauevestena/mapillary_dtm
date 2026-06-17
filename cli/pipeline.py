"""
Typer CLI orchestrator for the DTM-from-Mapillary pipeline.
"""

from __future__ import annotations

import logging
import json
import hashlib
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:  # Optional dependency for CLI usage
    import typer
except ImportError:  # pragma: no cover - typer is optional for library usage
    typer = None  # type: ignore[assignment]

from .. import constants
from ..api.mapillary_client import MapillaryClient
from ..common_core import FrameMeta, GroundPoint, ReconstructionResult
from ..depth.monodepth import CacheResult, predict_depths
from ..geom.anchors import find_anchors
from ..geom.height_solver import solve_scale_and_h
from ..geom.sfm_colmap import run as run_colmap
from ..geom.sfm_opensfm import run as run_opensfm
from ..geom.sfm_dim import run as run_dim
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
from ..ingest.imagery_cache import prefetch_imagery as cache_sequence_imagery
from ..io.writers import write_geotiffs, write_laz, write_ply_from_geotiff
from ..io.geojson_writers import write_frames_geojson, write_all_camera_positions_geojson
from ..osm.osmnx_utils import corridor_from_osm_bbox
from ..qa.qa_external import compare_to_geotiff
from ..qa.qa_internal import slope_from_plane_fit, write_agreement_maps
from ..qa.reports import write_html
from ..semantics.curb_edge_lane import extract_curbs_and_lanes
from ..semantics.ground_masks import prepare as prepare_masks
from ..fusion.heightmap_fusion import GridSpec, fuse as fuse_heightmap, _grid_from_points
from ..fusion.smoothing_regularization import edge_aware

log = logging.getLogger(__name__)

RUN_STAGES = [
    "ingestion",
    "preflight",
    "masks",
    "curbs",
    "opensfm",
    "colmap",
    "vo",
    "scale",
    "depth",
    "ground_extract",
    "consensus_breaklines_tin",
    "fusion_writers",
    "external_qa_report",
]


class _RunState:
    def __init__(
        self,
        path: Path,
        *,
        resume: bool = True,
        force_stages: Optional[list[str]] = None,
        inputs_fingerprint: Optional[str] = None,
    ) -> None:
        self.path = path
        self.resume = bool(resume)
        self.payload: dict[str, Any] = self._load() if self.resume else {}
        self.payload.setdefault("version", 1)
        self.payload.setdefault("created_at", _utc_now())
        self.payload["updated_at"] = _utc_now()
        self.payload.setdefault("stages", {})
        for stage in RUN_STAGES:
            self.payload["stages"].setdefault(stage, {"status": "pending"})
        if inputs_fingerprint:
            previous = self.payload.get("inputs_fingerprint")
            if previous and previous != inputs_fingerprint:
                for stage in RUN_STAGES:
                    self.payload["stages"][stage] = {
                        "status": "pending",
                        "invalidated_at": _utc_now(),
                        "reason": "inputs fingerprint changed",
                    }
            self.payload["inputs_fingerprint"] = inputs_fingerprint
        self.invalidate_from(force_stages or [])
        self.save()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf8"))
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:
            log.warning("Ignoring unreadable run state %s: %s", self.path, exc)
            return {}

    def stage(self, name: str) -> dict[str, Any]:
        _validate_stage_name(name)
        return self.payload.setdefault("stages", {}).setdefault(name, {"status": "pending"})

    def is_complete(self, name: str) -> bool:
        return self.resume and self.stage(name).get("status") == "complete"

    def start(self, name: str, *, inputs: Optional[dict[str, Any]] = None) -> None:
        record = self.stage(name)
        record.update(
            {
                "status": "running",
                "started_at": _utc_now(),
                "error": None,
            }
        )
        if inputs is not None:
            record["inputs"] = inputs
        self.save()

    def complete(
        self,
        name: str,
        *,
        outputs: Optional[dict[str, Any]] = None,
        counts: Optional[dict[str, Any]] = None,
        warnings: Optional[list[str]] = None,
    ) -> None:
        record = self.stage(name)
        record.update(
            {
                "status": "complete",
                "completed_at": _utc_now(),
                "error": None,
            }
        )
        if outputs is not None:
            record["outputs"] = outputs
        if counts is not None:
            record["counts"] = counts
        if warnings is not None:
            record["warnings"] = warnings
        self.save()

    def fail(self, name: str, exc: BaseException) -> None:
        record = self.stage(name)
        record.update(
            {
                "status": "failed",
                "failed_at": _utc_now(),
                "error": {
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                },
            }
        )
        self.save()

    def invalidate_from(self, force_stages: list[str]) -> list[str]:
        return _invalidate_forced_stages(self.payload, force_stages)

    def save(self) -> None:
        self.payload["updated_at"] = _utc_now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.payload, indent=2, default=_json_default), encoding="utf8")
        tmp.replace(self.path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_stage_name(name: str) -> None:
    if name not in RUN_STAGES:
        raise ValueError(f"Unknown stage '{name}'. Expected one of: {', '.join(RUN_STAGES)}")


def _invalidate_forced_stages(payload: dict[str, Any], force_stages: list[str]) -> list[str]:
    normalized = [str(stage).strip().lower() for stage in force_stages if str(stage).strip()]
    if not normalized:
        return []
    for stage in normalized:
        _validate_stage_name(stage)
    stages = payload.setdefault("stages", {})
    invalidated: list[str] = []
    first_index = min(RUN_STAGES.index(stage) for stage in normalized)
    reason = f"forced from {', '.join(normalized)}"
    for stage in RUN_STAGES[first_index:]:
        stages[stage] = {
            "status": "pending",
            "invalidated_at": _utc_now(),
            "reason": reason,
        }
        invalidated.append(stage)
    return invalidated


def _run_stage(
    run_state: _RunState,
    name: str,
    func,
    *,
    inputs: Optional[dict[str, Any]] = None,
    outputs=None,
    counts=None,
    warnings=None,
):
    run_state.start(name, inputs=inputs)
    try:
        result = func()
    except BaseException as exc:
        run_state.fail(name, exc)
        raise
    run_state.complete(
        name,
        outputs=outputs(result) if callable(outputs) else outputs,
        counts=counts(result) if callable(counts) else counts,
        warnings=warnings(result) if callable(warnings) else warnings,
    )
    return result


def run_pipeline(
    aoi_bbox: Optional[str] = None,
    out_dir: str = "./out",
    token: Optional[str] = None,
    dataset_dir: Optional[str] = None,
    imagery_root: Optional[str] = None,
    reference_dtm: Optional[str] = None,
    reference_nodata_values: Optional[str] = None,
    allow_synthetic: bool = False,
    strict_production: bool = True,
    use_learned_uncertainty: bool = False,
    uncertainty_model_path: Optional[str] = None,
    enforce_breaklines: bool = False,
    cache_imagery: bool = False,
    imagery_per_sequence: int = 5,
    colmap_threads: int = constants.COLMAP_DEFAULT_THREADS,
    colmap_use_gpu: bool = constants.COLMAP_USE_GPU,
    legacy_vo: bool = False,
    vo_force_synthetic: bool = False,
    vo_min_inliers: Optional[int] = None,
    resume: bool = True,
    force_stage: Optional[list[str]] = None,
    progress: Optional[bool] = None,
) -> dict:
    """
    Run the full pipeline over an AOI bbox or a local dataset bundle.
    Returns the manifest describing the run.

    Parameters
    ----------
    aoi_bbox : str
        Bounding box as "lon_min,lat_min,lon_max,lat_max"
    out_dir : str
        Output directory for results
    token : str, optional
        Mapillary API token (or use env var / file)
    dataset_dir : str, optional
        Local sample/dataset directory with config.json, sequences, and imagery.
    imagery_root : str, optional
        Root containing cached imagery laid out as <root>/<sequence>/<image>.jpg.
    reference_dtm : str, optional
        Held-out reference DTM used for external QA.
    reference_nodata_values : str, optional
        Comma-separated reference values to treat as nodata for external QA.
    allow_synthetic : bool
        Permit synthetic/heuristic fallbacks for development and smoke tests.
    strict_production : bool
        Fail when production prerequisites are missing instead of emitting
        plausible synthetic outputs.
    use_learned_uncertainty : bool
        Enable learned uncertainty calibration (default: False)
    uncertainty_model_path : str, optional
        Path to saved uncertainty model (will train if not found)
    enforce_breaklines : bool
        Enable breakline enforcement in TIN (default: False)
    cache_imagery : bool
        Prefetch Mapillary thumbnails into the local cache (default: False)
    imagery_per_sequence : int
        Maximum thumbnails to cache per retained sequence.
    colmap_threads : int
        Thread budget for COLMAP feature extraction / reconstruction.
    colmap_use_gpu : bool
        Enable COLMAP GPU paths (requires CUDA build).
    legacy_vo : bool
        Use the legacy OpenCV ORB VO pipeline instead of Deep-Image-Matching.
    vo_force_synthetic : bool
        Force the legacy synthetic VO path (skip OpenCV pipeline).
    vo_min_inliers : int, optional
        Minimum essential-matrix inliers required per frame pair.
    resume : bool
        Reuse completed stage artifacts when possible and keep run_state.json.
    force_stage : list[str], optional
        Stage names to invalidate, including downstream stages.
    progress : bool, optional
        Show tqdm progress bars. Defaults to interactive stderr only.

    Output Layout
    -------------
    The output directory (``out_dir``) has the following structure after a
    successful run::

        out_dir/
          manifest.json                  full run manifest (inputs, timings, outputs)
          run_state.json                 per-stage status used for --resume
          report.html                    QA summary report
          dtm_0p5m_ellipsoid.tif         primary DTM raster (0.5 m, ellipsoidal heights)
          slope_deg.tif                  slope raster (degrees)
          confidence.tif                 confidence raster [0..1]
          ground_points.laz              consensus ground points (LAS/LAZ point cloud)
          dtm_0p5m_ellipsoid.ply         PLY mesh converted from the DTM raster
          metadata/
            frames.geojson               all ingested frames — GNSS camera position
                                         (written after the ingestion stage)
            camera_positions.geojson     SfM-refined camera positions in WGS84,
                                         one feature per recovered pose, tagged with
                                         source (opensfm / colmap / vo)
                                         (written after the fusion/writers stage)
          cache/
            opensfm/                     OpenSfM workspace
            colmap/                      COLMAP workspace
            masks/                       ground segmentation masks (.npz per frame)
            depth_mono/                  monocular depth maps (.npz per frame)
          qa/
            qa_summary.json              QA metrics summary
            agreement_maps.npz           per-source DTM agreement maps

    The ``manifest.json`` ``outputs`` key lists the absolute paths to every
    generated file so downstream tools can locate them without hard-coding
    directory assumptions.
    """
    if strict_production and allow_synthetic:
        raise ValueError("--allow-synthetic cannot be combined with strict production mode")

    timings: dict[str, float] = {}
    run_warnings: list[str] = []
    run_start = time.perf_counter()
    os.makedirs(out_dir, exist_ok=True)
    out_path = Path(out_dir)
    qa_dir = out_path / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = out_path / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    mask_cache_dir = out_path / "cache" / "masks"
    depth_cache_dir = out_path / "cache" / "depth_mono"
    progress_enabled = _progress_enabled(progress)
    run_state = _RunState(
        out_path / "run_state.json",
        resume=resume,
        force_stages=force_stage or [],
    )
    if resume and run_state.is_complete("external_qa_report") and _final_outputs_valid(out_path):
        return json.loads((out_path / "manifest.json").read_text(encoding="utf8"))
    dataset_path = Path(dataset_dir) if dataset_dir else None
    bbox = _resolve_bbox(aoi_bbox, dataset_path)
    imagery_root_path = _resolve_imagery_root(dataset_path, imagery_root)

    # Paths for GeoJSON metadata outputs
    frames_geojson_path = metadata_dir / "frames.geojson"
    camera_positions_geojson_path = metadata_dir / "camera_positions.geojson"

    map_client: MapillaryClient | None = None
    t0 = time.perf_counter()
    def _ingest():
        nonlocal map_client
        if dataset_path is not None:
            loaded = _load_dataset_sequences(dataset_path, bbox)
            if not loaded:
                raise RuntimeError(f"No usable sequence metadata found under {dataset_path}")
            return loaded
        map_client = MapillaryClient(token=token)
        loaded = discover_sequences(bbox, token=token, client=map_client)
        return filter_car_sequences(loaded)

    seqs = _run_stage(
        run_state,
        "ingestion",
        _ingest,
        inputs={
            "dataset_dir": str(dataset_path) if dataset_path else None,
            "bbox": bbox,
        },
        counts=lambda loaded: _sequence_counts(loaded),
        outputs=lambda loaded: {"frames_geojson": str(frames_geojson_path)},
    )
    # Write frame metadata as GeoJSON immediately after ingestion so the GNSS
    # camera positions are available for inspection before reconstruction runs.
    try:
        write_frames_geojson(seqs, frames_geojson_path)
    except Exception as exc:
        log.warning("Failed to write frames GeoJSON: %s", exc)
        run_warnings.append(f"frames GeoJSON write failed: {exc}")

    input_fingerprint = _run_inputs_fingerprint(
        seqs,
        bbox=bbox,
        dataset_dir=dataset_path,
        imagery_root=imagery_root_path,
        reference_dtm=reference_dtm,
        strict_production=strict_production,
        allow_synthetic=allow_synthetic,
    )
    previous_fingerprint = run_state.payload.get("inputs_fingerprint")
    if previous_fingerprint and previous_fingerprint != input_fingerprint:
        run_state.invalidate_from(["ingestion"])
        run_state.complete("ingestion", counts=_sequence_counts(seqs))
    run_state.payload["inputs_fingerprint"] = input_fingerprint
    run_state.save()
    timings["ingestion_s"] = time.perf_counter() - t0

    preflight = {}
    t0 = time.perf_counter()
    if strict_production:
        preflight = _run_stage(
            run_state,
            "preflight",
            lambda: _strict_preflight(
                seqs,
                imagery_root_path=imagery_root_path,
                reference_dtm=reference_dtm,
                vo_force_synthetic=vo_force_synthetic,
            ),
            inputs={"strict_production": strict_production},
        )
    else:
        run_state.complete(
            "preflight",
            counts={"strict_production": False},
            warnings=["strict production preflight skipped"],
        )
    timings["preflight_s"] = time.perf_counter() - t0

    imagery_cache_stats = {}
    if cache_imagery:
        if map_client is None:
            map_client = MapillaryClient(token=token)
        imagery_cache_stats = cache_sequence_imagery(
            seqs,
            client=map_client,
            cache_dir=imagery_root_path,
            max_per_sequence=imagery_per_sequence if imagery_per_sequence > 0 else None,
        )
        if imagery_cache_stats:
            log.info(
                "Cached thumbnails for %d sequences (%d images)",
                len(imagery_cache_stats),
                sum(imagery_cache_stats.values()),
            )
    t0 = time.perf_counter()
    mask_index = _run_stage(
        run_state,
        "masks",
        lambda: prepare_masks(
            seqs,
            out_dir=mask_cache_dir,
            imagery_root=imagery_root_path,
            allow_heuristic=allow_synthetic,
            progress=progress_enabled,
        ),
        inputs={
            "cache_dir": str(mask_cache_dir),
            "allow_heuristic": allow_synthetic,
            "resume": resume,
        },
        outputs=lambda result: {"cache_dir": str(mask_cache_dir)},
        counts=lambda result: _mask_cache_counts(seqs, mask_cache_dir, strict=not allow_synthetic),
    )
    timings["masks_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    curb_lines = _run_stage(
        run_state,
        "curbs",
        lambda: extract_curbs_and_lanes(seqs),
        counts=lambda result: {
            "sequences": len(result),
            "lines": int(sum(len(lines) for lines in result.values())),
        },
    )
    timings["curbs_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    reconA = _run_stage(
        run_state,
        "opensfm",
        lambda: run_opensfm(
            seqs,
            imagery_root=imagery_root_path,
            workspace_root=out_path / "cache" / "opensfm",
            allow_synthetic=allow_synthetic,
            progress=progress_enabled,
        ),
        inputs={"workspace_root": str(out_path / "cache" / "opensfm")},
        counts=lambda result: _reconstruction_counts(result),
    )
    timings["opensfm_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    if not legacy_vo and constants.DIM_ENABLED:
        reconB = _run_stage(
            run_state,
            "colmap",
            lambda: run_dim(
                seqs,
                imagery_root=imagery_root_path,
                workspace_root=out_path / "cache" / "colmap",
                progress=progress_enabled,
            ),
            inputs={
                "workspace_root": str(out_path / "cache" / "colmap"),
            },
            counts=lambda result: _reconstruction_counts(result),
        )
        if not allow_synthetic and not reconB:
            from ..geom.colmap_adapter import COLMAPUnavailable
            raise COLMAPUnavailable("COLMAP synthetic fallback disabled and no real reconstruction was produced")
    else:
        reconB = _run_stage(
            run_state,
            "colmap",
            lambda: run_colmap(
                seqs,
                threads=colmap_threads,
                use_gpu=colmap_use_gpu,
                workspace_root=out_path / "cache" / "colmap",
                imagery_root=imagery_root_path,
                allow_synthetic=allow_synthetic,
                progress=progress_enabled,
            ),
            inputs={
                "workspace_root": str(out_path / "cache" / "colmap"),
                "threads": colmap_threads,
                "use_gpu": colmap_use_gpu,
            },
            counts=lambda result: _reconstruction_counts(result),
        )
    timings["colmap_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    t0 = time.perf_counter()
    if not legacy_vo and constants.DIM_ENABLED:
        vo = _run_stage(
            run_state,
            "vo",
            lambda: run_dim(
                seqs,
                imagery_root=imagery_root_path,
                progress=progress_enabled,
            ),
            counts=lambda result: _reconstruction_counts(result),
        )
    else:
        vo = _run_stage(
            run_state,
            "vo",
            lambda: run_vo(
                seqs,
                imagery_root=imagery_root_path,
                force_synthetic=vo_force_synthetic,
                min_inliers=vo_min_inliers,
                allow_synthetic=allow_synthetic,
                progress=progress_enabled,
            ),
            inputs={"force_synthetic": vo_force_synthetic, "min_inliers": vo_min_inliers},
            counts=lambda result: _reconstruction_counts(result),
        )
    timings["vo_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    anchors, scales, heights = _run_stage(
        run_state,
        "scale",
        lambda: (
            lambda found_anchors: (
                found_anchors,
                *solve_scale_and_h(reconA, reconB, vo, found_anchors, seqs),
            )
        )(find_anchors(seqs, token=token, allow_synthetic=allow_synthetic)),
        counts=lambda result: {
            "anchors": len(result[0]) if result[0] is not None else 0,
            "scales": len(result[1]) if result[1] is not None else 0,
        },
    )
    timings["scale_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    mono_depths: CacheResult = _run_stage(
        run_state,
        "depth",
        lambda: predict_depths(
            seqs,
            out_dir=depth_cache_dir,
            imagery_root=imagery_root_path,
            allow_synthetic=allow_synthetic,
            progress=progress_enabled,
        ),
        inputs={
            "cache_dir": str(depth_cache_dir),
            "allow_synthetic": allow_synthetic,
            "resume": resume,
        },
        outputs=lambda result: {"cache_dir": str(depth_cache_dir)},
        counts=lambda result: _depth_cache_counts(seqs, depth_cache_dir, strict=not allow_synthetic),
    )
    timings["depth_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    def _extract_ground():
        extracted: dict[str, list[GroundPoint]] = {}
        sources = [
            ("opensfm", reconA, {"include_sparse": True, "include_monodepth": True}),
            ("colmap", reconB, {"include_sparse": True, "include_monodepth": True}),
            ("vo", vo, {"include_sparse": False, "include_monodepth": False}),
        ]
        for name, recon, options in _progress_iter(sources, "Ground extraction", progress_enabled):
            extracted[name] = label_and_filter_points(
                recon,
                scales,
                mask_dir=mask_cache_dir,
                mono_cache=depth_cache_dir,
                vo_recon=vo,
                imagery_root=imagery_root_path,
                include_plane_sweep=allow_synthetic if name != "vo" else False,
                allow_synthetic_depth=allow_synthetic,
                mono_depths=mono_depths if options["include_monodepth"] else None,
                **options,
            )
        return extracted

    extracted_points = _run_stage(
        run_state,
        "ground_extract",
        _extract_ground,
        counts=lambda result: {name: len(points) for name, points in result.items()},
    )
    ptsA = extracted_points.get("opensfm", [])
    ptsB = extracted_points.get("colmap", [])
    ptsC = extracted_points.get("vo", [])
    timings["ground_extract_s"] = time.perf_counter() - t0

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

    t0 = time.perf_counter()
    run_state.start("consensus_breaklines_tin")
    consensus = consensus_agree(ptsA, ptsB, ptsC)

    # Breakline processing (if enabled)
    breakline_stats = {
        "enabled": enforce_breaklines,
        "curbs_detected": sum(len(lines) for lines in curb_lines.values()) if curb_lines else 0,
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
            camera_poses, camera_models = _camera_pose_and_model_dicts(reconA)

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
            if (
                enforce_breaklines
                and breakline_vertices is not None
                and breakline_edges
            ):
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
    run_state.complete(
        "consensus_breaklines_tin",
        counts={
            "consensus": len(consensus),
            "tin_samples": len(tin_samples),
            "exported_candidates": len(consensus_all),
            "breakline_vertices": int(breakline_stats.get("vertices", 0)),
            "breakline_edges": int(breakline_stats.get("edges", 0)),
        },
    )
    timings["consensus_breaklines_tin_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    run_state.start("fusion_writers")
    # Compute the grid from the consensus points *before* TIN augmentation so
    # that TIN-extrapolated samples (which can extend far beyond actual data
    # coverage) do not inflate the bounding box.  The same grid is reused for
    # the fused DTM raster, the per-source agreement maps, and the geotiff
    # transform, guaranteeing that generation and evaluation share identical
    # cell boundaries.
    consensus_grid = _grid_from_points(consensus)
    dtm, conf, grid = fuse_heightmap(consensus_all, return_grid=True, grid=consensus_grid)
    dtm_s = edge_aware(dtm)
    slope_deg, aspect = slope_from_plane_fit(dtm_s)

    lon0, lat0, h0 = _infer_origin(seqs, bbox)
    transform, raster_crs = _grid_transform_webmercator(grid, lon0=lon0, lat0=lat0)
    dtm_write = np.flipud(dtm_s)
    slope_write = np.flipud(slope_deg)
    conf_write = np.flipud(conf)
    geotiff_paths = write_geotiffs(
        out_dir, dtm_write, slope_write, conf_write, transform=transform, crs=raster_crs
    )
    # Generate PLY file from the DTM geotiff
    ply_path = write_ply_from_geotiff(geotiff_paths["dtm_0p5m_ellipsoid.tif"], out_dir)
    laz_points, laz_attrs = _points_and_attrs_from_consensus(consensus_all)
    laz_path = write_laz(out_dir, laz_points, attrs=laz_attrs)

    agreement_path = qa_dir / "agreement_maps.npz"
    source_dtms = _source_dtms_on_grid({"opensfm": ptsA, "colmap": ptsB, "vo": ptsC}, grid)
    agreement_results = write_agreement_maps(agreement_path, dtm_s, source_dtms)
    agreement_summary = _summarize_agreement(agreement_results)

    # Write SfM-refined camera positions from all three sources combined so the
    # user can overlay GNSS (frames.geojson) vs. SfM positions (camera_positions.geojson)
    # directly in a GIS tool without any further post-processing.
    try:
        write_all_camera_positions_geojson(
            {"opensfm": reconA, "colmap": reconB, "vo": vo},
            lon0, lat0, h0,
            camera_positions_geojson_path,
        )
    except Exception as exc:
        log.warning("Failed to write camera positions GeoJSON: %s", exc)
        run_warnings.append(f"camera_positions GeoJSON write failed: {exc}")

    run_state.complete(
        "fusion_writers",
        outputs={
            "geotiffs": geotiff_paths,
            "laz": laz_path,
            "agreement_maps": str(agreement_path),
            "frames_geojson": str(frames_geojson_path),
            "camera_positions_geojson": str(camera_positions_geojson_path),
        },
        counts={
            "dtm_shape": list(dtm_s.shape),
            "ground_points": int(laz_points.shape[0]),
        },
    )
    timings["fusion_writers_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    run_state.start("external_qa_report")
    external_stats = None
    dtm_key = "dtm_0p5m_ellipsoid.tif"
    qa_reference = Path(reference_dtm) if reference_dtm else None
    if dtm_key in geotiff_paths and qa_reference and qa_reference.exists():
        external_stats = compare_to_geotiff(
            geotiff_paths[dtm_key],
            str(qa_reference),
            out_dir=qa_dir,
            reference_nodata_values=_parse_float_list(reference_nodata_values),
        )
    elif reference_dtm:
        raise FileNotFoundError(f"Reference DTM not found: {reference_dtm}")
    else:
        run_warnings.append("No reference DTM supplied; external accuracy QA skipped")
    qa_status = "complete" if external_stats else "incomplete"

    manifest = {
        "bbox": bbox,
        "dataset": _dataset_manifest(dataset_path),
        "synthetic_policy": {
            "allow_synthetic": allow_synthetic,
            "strict_production": strict_production,
        },
        "provenance": {
            "git_sha": _git_sha(),
            "token_hash": _token_hash(map_client.token if map_client else token),
            "models": _model_provenance_snapshot(),
        },
        "preflight": preflight,
        "timings_s": timings,
        "warnings": run_warnings,
        "scales": {k: float(v) for k, v in (scales or {}).items()},
        "heights": {k: float(v) for k, v in (heights or {}).items()},
        "corridor_source": corridor_info.get("source") if corridor_info else None,
        "corridor_buffer_m": corridor_info.get("buffer_m") if corridor_info else None,
        "tin_samples": len(tin_samples),
        "ingestion": {
            "sequence_count": len(seqs),
            "image_count": int(sum(len(frames) for frames in seqs.values())),
            "imagery_cache": {
                "enabled": cache_imagery,
                "sequences": len(imagery_cache_stats) if imagery_cache_stats else 0,
                "images": int(sum(imagery_cache_stats.values())) if imagery_cache_stats else 0,
                "cache_root": constants.MAPILLARY_CACHE_ROOT,
                "imagery_root": str(imagery_root_path) if imagery_root_path else None,
                "mask_cache": str(mask_cache_dir),
                "depth_cache": str(depth_cache_dir),
            },
            "mask_count": int(sum(len(paths) for paths in mask_index.values())),
        },
        "reconstruction": {
            "opensfm": _summarize_reconstruction_sources(reconA),
            "colmap": {
                "threads": colmap_threads,
                "use_gpu": colmap_use_gpu,
                "sources": _summarize_reconstruction_sources(reconB),
            },
            "vo": {
                "force_synthetic": vo_force_synthetic,
                "min_inliers": int(vo_min_inliers)
                if vo_min_inliers is not None
                else constants.VO_MIN_INLIERS,
                "sources": _summarize_reconstruction_sources(vo),
            },
        },
        "ground_points": {
            "opensfm": len(ptsA),
            "colmap": len(ptsB),
            "vo": len(ptsC),
            "consensus": len(consensus),
            "exported": int(laz_points.shape[0]),
        },
        "breaklines": breakline_stats,
        "outputs": {
            "geotiffs": geotiff_paths,
            "laz": laz_path,
            "agreement_maps": str(agreement_path),
            "frames_geojson": str(frames_geojson_path),
            "camera_positions_geojson": str(camera_positions_geojson_path),
            "manifest": str(out_path / "manifest.json"),
            "qa_summary": str(qa_dir / "qa_summary.json"),
            "run_state": str(out_path / "run_state.json"),
        },
        "qa": {
            "status": qa_status,
            "policy": "report-only",
            "agreement_summary": agreement_summary,
            "external": external_stats,
            "reference_dtm": str(qa_reference) if qa_reference else None,
            "reference_nodata_values": _parse_float_list(reference_nodata_values),
        },
        "constants": _constants_snapshot(),
    }
    timings["total_s"] = time.perf_counter() - run_start
    qa_metrics = {}
    qa_metrics.update({f"agreement_{k}": v for k, v in agreement_summary.items()})
    if external_stats:
        qa_metrics.update({f"external_{k}": v for k, v in external_stats.items()})
    qa_summary = {
        "status": qa_status,
        "policy": "report-only",
        "agreement": agreement_summary,
        "external": external_stats,
        "warnings": run_warnings,
    }
    _write_json(out_path / "manifest.json", manifest)
    _write_json(qa_dir / "qa_summary.json", qa_summary)

    artifact_paths = {
        "DTM": geotiff_paths.get(dtm_key, ""),
        "Slope": geotiff_paths.get("slope_deg.tif", ""),
        "Confidence": geotiff_paths.get("confidence.tif", ""),
        "LAZ / NPZ": laz_path,
        "Agreement maps": str(agreement_path),
        "Frame positions (GeoJSON)": str(frames_geojson_path),
        "Camera positions (GeoJSON)": str(camera_positions_geojson_path),
        "Manifest": str(out_path / "manifest.json"),
        "QA summary": str(qa_dir / "qa_summary.json"),
    }
    write_html(out_dir, manifest, qa_summary=qa_metrics, artifact_paths=artifact_paths)
    run_state.complete(
        "external_qa_report",
        outputs={
            "manifest": str(out_path / "manifest.json"),
            "qa_summary": str(qa_dir / "qa_summary.json"),
            "report": str(out_path / "report.html"),
        },
        counts={"qa_status": qa_status, "external_n": external_stats.get("n") if external_stats else 0},
        warnings=run_warnings,
    )
    timings["external_qa_report_s"] = time.perf_counter() - t0
    return manifest


# Backwards-compatible alias for library callers.
run = run_pipeline

if typer is not None:
    app = typer.Typer(help="DTM from Mapillary — high-accuracy pipeline")

    @app.callback()
    def cli_main() -> None:
        """DTM from Mapillary command group."""

    @app.command("run")
    def cli_run(
        aoi_bbox_arg: Optional[str] = typer.Argument(
            None,
            help="Optional bbox as lon_min,lat_min,lon_max,lat_max. Prefer --aoi-bbox for scripts.",
        ),
        aoi_bbox: Optional[str] = typer.Option(
            None,
            "--aoi-bbox",
            help="Area of interest as lon_min,lat_min,lon_max,lat_max.",
        ),
        out_dir: str = "./out",
        token: Optional[str] = None,
        dataset_dir: Optional[str] = typer.Option(
            None,
            "--dataset-dir",
            help="Local dataset bundle root containing config.json, sequences, and imagery.",
        ),
        imagery_root: Optional[str] = typer.Option(
            None,
            "--imagery-root",
            help="Cached imagery root laid out as <root>/<sequence>/<image>.jpg.",
        ),
        reference_dtm: Optional[str] = typer.Option(
            None,
            "--reference-dtm",
            help="Held-out reference DTM GeoTIFF for external QA.",
        ),
        reference_nodata_values: Optional[str] = typer.Option(
            None,
            "--reference-nodata-values",
            help="Comma-separated reference values to treat as nodata for external QA.",
        ),
        allow_synthetic: bool = typer.Option(
            False,
            help="Permit synthetic/heuristic fallbacks for development or smoke tests.",
        ),
        strict_production: bool = typer.Option(
            True,
            help="Fail when production prerequisites are missing.",
        ),
        use_learned_uncertainty: bool = False,
        uncertainty_model_path: Optional[str] = None,
        enforce_breaklines: bool = False,
        cache_imagery: bool = False,
        imagery_per_sequence: int = 5,
        colmap_threads: int = typer.Option(
            constants.COLMAP_DEFAULT_THREADS,
            help="Thread budget for COLMAP feature extraction / mapping.",
        ),
        colmap_use_gpu: bool = typer.Option(
            constants.COLMAP_USE_GPU,
            help="Enable COLMAP GPU paths (requires CUDA-enabled build).",
        ),
        legacy_vo: bool = typer.Option(
            False,
            help="Use the legacy OpenCV ORB VO pipeline instead of Deep-Image-Matching.",
        ),
        vo_force_synthetic: bool = typer.Option(
            False,
            help="Force the legacy synthetic VO path (skip imagery-backed VO).",
        ),
        vo_min_inliers: Optional[int] = typer.Option(
            None,
            help="Minimum VO inliers required per frame pair (default: constants.VO_MIN_INLIERS).",
        ),
        resume: bool = typer.Option(
            True,
            "--resume/--no-resume",
            help="Reuse completed stage artifacts and maintain run_state.json.",
        ),
        force_stage: Optional[list[str]] = typer.Option(
            None,
            "--force-stage",
            help=f"Invalidate a stage and downstream stages. Choices: {', '.join(RUN_STAGES)}.",
        ),
        progress: Optional[bool] = typer.Option(
            None,
            "--progress/--no-progress",
            help="Show tqdm progress bars. Defaults to interactive stderr only.",
        ),
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
        cache_imagery : bool
            Prefetch Mapillary thumbnails into cache
        imagery_per_sequence : int
            How many thumbnails to cache per sequence (when enabled)
        colmap_threads : int
            Thread budget for COLMAP feature extraction / reconstruction
        colmap_use_gpu : bool
            Enable COLMAP GPU pipelines (requires CUDA build)
        vo_force_synthetic : bool
            Force the synthetic VO fallback path
        vo_min_inliers : int, optional
            Minimum VO inliers per frame pair
        """
        run_pipeline(
            aoi_bbox=aoi_bbox or aoi_bbox_arg,
            out_dir=out_dir,
            token=token,
            dataset_dir=dataset_dir,
            imagery_root=imagery_root,
            reference_dtm=reference_dtm,
            reference_nodata_values=reference_nodata_values,
            allow_synthetic=allow_synthetic,
            strict_production=strict_production,
            use_learned_uncertainty=use_learned_uncertainty,
            uncertainty_model_path=uncertainty_model_path,
            enforce_breaklines=enforce_breaklines,
            cache_imagery=cache_imagery,
            imagery_per_sequence=imagery_per_sequence,
            colmap_threads=colmap_threads,
            colmap_use_gpu=colmap_use_gpu,
            legacy_vo=legacy_vo,
            vo_force_synthetic=vo_force_synthetic,
            vo_min_inliers=vo_min_inliers,
            resume=resume,
            force_stage=force_stage,
            progress=progress,
        )

else:  # pragma: no cover - only executed when typer missing
    app = None


def _resolve_bbox(aoi_bbox: Optional[str], dataset_dir: Optional[Path]) -> tuple[float, float, float, float]:
    if aoi_bbox:
        coords = tuple(float(x.strip()) for x in aoi_bbox.split(","))
        if len(coords) != 4:
            raise ValueError("aoi_bbox must be lon_min,lat_min,lon_max,lat_max")
        return coords
    if dataset_dir is not None:
        config_path = dataset_dir / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf8"))
            if config.get("bbox_string"):
                return _resolve_bbox(str(config["bbox_string"]), None)
            bbox = config.get("bbox")
            if isinstance(bbox, dict):
                return (
                    float(bbox["min_lon"]),
                    float(bbox["min_lat"]),
                    float(bbox["max_lon"]),
                    float(bbox["max_lat"]),
                )
    bbox = constants.bbox
    return (
        float(bbox["min_lon"]),
        float(bbox["min_lat"]),
        float(bbox["max_lon"]),
        float(bbox["max_lat"]),
    )


def _resolve_imagery_root(dataset_dir: Optional[Path], imagery_root: Optional[str]) -> Optional[Path]:
    if imagery_root:
        return Path(imagery_root)
    if dataset_dir is not None and (dataset_dir / "imagery").exists():
        return dataset_dir / "imagery"
    return None


def _progress_enabled(progress: Optional[bool]) -> bool:
    if progress is not None:
        return bool(progress)
    try:
        return sys.stderr.isatty()
    except Exception:
        return False


def _progress_iter(items, desc: str, enabled: bool):
    if not enabled:
        return items
    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover - optional display dependency
        return items
    return tqdm(items, desc=desc, unit="item")


def _sequence_counts(seqs: dict[str, list[FrameMeta]]) -> dict[str, int]:
    return {
        "sequences": len(seqs),
        "images": int(sum(len(frames) for frames in seqs.values())),
    }


def _run_inputs_fingerprint(
    seqs: dict[str, list[FrameMeta]],
    *,
    bbox: tuple[float, float, float, float],
    dataset_dir: Optional[Path],
    imagery_root: Optional[Path],
    reference_dtm: Optional[str],
    strict_production: bool,
    allow_synthetic: bool,
) -> str:
    frames = [
        {
            "seq_id": frame.seq_id,
            "image_id": frame.image_id,
            "captured_at_ms": int(frame.captured_at_ms),
            "lon": round(float(frame.lon), 9),
            "lat": round(float(frame.lat), 9),
        }
        for seq_id in sorted(seqs)
        for frame in seqs[seq_id]
    ]
    payload = {
        "bbox": bbox,
        "dataset_dir": str(dataset_dir) if dataset_dir else None,
        "imagery_root": str(imagery_root) if imagery_root else None,
        "reference_dtm": str(reference_dtm) if reference_dtm else None,
        "strict_production": strict_production,
        "allow_synthetic": allow_synthetic,
        "frames": frames,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(raw.encode("utf8")).hexdigest()


def _expected_cache_paths(seqs: dict[str, list[FrameMeta]], cache_dir: Path) -> list[Path]:
    return [cache_dir / f"{frame.image_id}.npz" for frames in seqs.values() for frame in frames]


def _mask_cache_counts(
    seqs: dict[str, list[FrameMeta]],
    cache_dir: Path,
    *,
    strict: bool,
) -> dict[str, object]:
    return _npz_cache_counts(
        _expected_cache_paths(seqs, cache_dir),
        required_keys=("prob",),
        strict=strict,
    )


def _depth_cache_counts(
    seqs: dict[str, list[FrameMeta]],
    cache_dir: Path,
    *,
    strict: bool,
) -> dict[str, object]:
    return _npz_cache_counts(
        _expected_cache_paths(seqs, cache_dir),
        required_keys=("depth", "uncertainty"),
        strict=strict,
    )


def _validate_mask_cache(seqs: dict[str, list[FrameMeta]], cache_dir: Path, *, strict: bool) -> bool:
    counts = _mask_cache_counts(seqs, cache_dir, strict=strict)
    return counts["expected"] == counts["valid"]


def _validate_depth_cache(seqs: dict[str, list[FrameMeta]], cache_dir: Path, *, strict: bool) -> bool:
    counts = _depth_cache_counts(seqs, cache_dir, strict=strict)
    return counts["expected"] == counts["valid"]


def _npz_cache_counts(
    paths: list[Path],
    *,
    required_keys: tuple[str, ...],
    strict: bool,
) -> dict[str, object]:
    present = 0
    valid = 0
    invalid: list[str] = []
    source_types: dict[str, int] = {}
    for path in paths:
        if not path.exists():
            invalid.append(str(path))
            continue
        present += 1
        try:
            with np.load(path, allow_pickle=False) as data:
                missing = [key for key in required_keys if key not in data.files]
                source_type = _npz_scalar(data, "source_type") or ""
                source_types[source_type or "missing"] = source_types.get(source_type or "missing", 0) + 1
                if missing:
                    invalid.append(str(path))
                    continue
                if strict and source_type not in {"model", "external"}:
                    invalid.append(str(path))
                    continue
                valid += 1
        except Exception:
            invalid.append(str(path))
    return {
        "expected": len(paths),
        "present": present,
        "valid": valid,
        "invalid": len(invalid),
        "invalid_examples": invalid[:10],
        "source_types": source_types,
        "strict": strict,
    }


def _npz_scalar(data, key: str) -> str | None:
    if key not in data:
        return None
    value = data[key]
    if getattr(value, "shape", ()) == ():
        raw = value.item()
    else:
        raw = value
    if raw is None:
        return None
    text = str(raw)
    return text if text and text != "None" else None


def _reconstruction_counts(recon: dict[str, ReconstructionResult]) -> dict[str, int]:
    return {
        "sequences": len(recon),
        "frames": int(sum(len(result.frames) for result in recon.values())),
        "poses": int(sum(len(result.poses) for result in recon.values())),
        "points": int(sum(int(result.points_xyz.shape[0]) for result in recon.values())),
    }


def _final_outputs_valid(out_path: Path) -> bool:
    expected = [
        out_path / "dtm_0p5m_ellipsoid.tif",
        out_path / "slope_deg.tif",
        out_path / "confidence.tif",
        out_path / "manifest.json",
        out_path / "qa" / "qa_summary.json",
        out_path / "report.html",
    ]
    return all(path.exists() for path in expected)


def _strict_preflight(
    seqs: dict[str, list[FrameMeta]],
    *,
    imagery_root_path: Optional[Path],
    reference_dtm: Optional[str],
    vo_force_synthetic: bool,
) -> dict[str, object]:
    checks: list[dict[str, object]] = []

    def add(name: str, ok: bool, message: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "message": message})

    forced_envs = [
        name
        for name in ("OPEN_SFM_FORCE_SYNTHETIC", "COLMAP_FORCE_SYNTHETIC")
        if _env_truthy(name)
    ]
    add(
        "synthetic env",
        not forced_envs and not vo_force_synthetic,
        "no forced synthetic flags set"
        if not forced_envs and not vo_force_synthetic
        else f"forced synthetic flags present: {forced_envs}, vo_force_synthetic={vo_force_synthetic}",
    )

    if imagery_root_path is None or not imagery_root_path.exists():
        add("imagery root", False, f"imagery root missing: {imagery_root_path}")
    else:
        image_paths = [_frame_image_path(frame, imagery_root_path) for frames in seqs.values() for frame in frames]
        missing = sum(1 for path in image_paths if path is None)
        readable_errors = _sample_readable_errors([path for path in image_paths if path is not None])
        add(
            "imagery files",
            missing == 0 and not readable_errors,
            f"{len(image_paths) - missing}/{len(image_paths)} images found; readable sample ok"
            if missing == 0 and not readable_errors
            else f"missing={missing}, unreadable_sample={readable_errors[:5]}",
        )

    if reference_dtm:
        ref_path = Path(reference_dtm)
        add("reference DTM", ref_path.exists(), f"found {ref_path}" if ref_path.exists() else f"missing {ref_path}")

    colmap_bin = os.getenv("COLMAP_BIN") or "colmap"
    colmap_ok = shutil.which(colmap_bin) is not None or _docker_image_available(os.getenv("COLMAP_DOCKER_IMAGE"))
    add(
        "COLMAP",
        colmap_ok,
        f"binary {colmap_bin} or Docker image available"
        if colmap_ok
        else "COLMAP binary missing and COLMAP_DOCKER_IMAGE unavailable",
    )

    opensfm_bin = os.getenv("OPEN_SFM_BIN") or "opensfm_run_all"
    opensfm_image = os.getenv("OPEN_SFM_DOCKER_IMAGE", "freakthemighty/opensfm:latest")
    opensfm_ok = shutil.which(opensfm_bin) is not None or _docker_image_available(opensfm_image)
    add(
        "OpenSfM",
        opensfm_ok,
        f"binary {opensfm_bin} or Docker image {opensfm_image} available"
        if opensfm_ok
        else f"OpenSfM binary missing and Docker image {opensfm_image} unavailable",
    )

    ground_model_path = os.getenv("GROUND_MASK_MODEL_PATH")
    ground_model_ok = bool(ground_model_path and Path(ground_model_path).exists()) or _hf_model_cached(
        os.getenv("GROUND_MASK_MODEL_ID", "nvidia/segformer-b0-finetuned-cityscapes-512-1024")
    )
    add(
        "ground mask model",
        ground_model_ok,
        "TorchScript path or cached Hugging Face model available"
        if ground_model_ok
        else "missing GROUND_MASK_MODEL_PATH and cached SegFormer ground model",
    )

    depth_model_path = os.getenv("MONODEPTH_MODEL_PATH")
    depth_model_ok = bool(depth_model_path and Path(depth_model_path).exists()) or _hf_model_cached(
        os.getenv("MONODEPTH_MODEL_ID", "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf")
    )
    add(
        "monodepth model",
        depth_model_ok,
        "TorchScript path or cached Hugging Face model available"
        if depth_model_ok
        else "missing MONODEPTH_MODEL_PATH and cached Depth Anything model",
    )

    errors = [check for check in checks if not check["ok"]]
    result = {"status": "pass" if not errors else "fail", "checks": checks}
    if errors:
        message = "; ".join(f"{item['name']}: {item['message']}" for item in errors)
        raise RuntimeError(f"Strict production preflight failed: {message}")
    return result


def _env_truthy(name: str) -> bool:
    value = os.getenv(name)
    return bool(value and value.lower() not in {"0", "false", "no", "off"})


def _docker_image_available(image: str | None) -> bool:
    if not image or shutil.which("docker") is None:
        return False
    try:
        subprocess.run(
            ["docker", "image", "inspect", image],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return True
    except Exception:
        return False


def _hf_model_cached(model_id: str | None) -> bool:
    if not model_id:
        return False
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return False
    try:
        path = try_to_load_from_cache(
            model_id,
            "config.json",
            cache_dir=os.getenv("DTM_MODEL_CACHE_DIR", "models/huggingface"),
            revision=os.getenv("DTM_MODEL_REVISION"),
        )
    except Exception:
        return False
    if path is None or not isinstance(path, (str, os.PathLike)):
        return False
    return Path(path).exists()


def _frame_image_path(frame: FrameMeta, imagery_root: Path) -> Path | None:
    from ..ingest.image_loader import ImageryLoader
    loader = ImageryLoader(imagery_root)
    seq_dir = loader._sequence_dir(frame.seq_id)
    for path in loader._candidate_paths(seq_dir, frame.image_id):
        if path.exists():
            return path
    return None


def _sample_readable_errors(paths: list[Path], *, limit: int = 30) -> list[str]:
    errors: list[str] = []
    for path in paths[:limit]:
        try:
            import cv2  # type: ignore

            if cv2.imread(str(path), cv2.IMREAD_UNCHANGED) is None:
                errors.append(str(path))
        except Exception:
            try:
                from PIL import Image

                with Image.open(path) as image:
                    image.verify()
            except Exception:
                errors.append(str(path))
    return errors


def _load_dataset_sequences(
    dataset_dir: Path,
    bbox: tuple[float, float, float, float],
) -> dict[str, list[FrameMeta]]:
    metadata_path = dataset_dir / "sequences" / "metadata.geojson"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found: {metadata_path}")
    try:
        import geopandas as gpd
    except ImportError as exc:  # pragma: no cover - requirements include geopandas
        raise RuntimeError("geopandas is required to load dataset metadata") from exc

    kept_ids = _dataset_kept_sequence_ids(dataset_dir)
    gdf = gpd.read_file(metadata_path)
    sequences: dict[str, list[FrameMeta]] = {}
    lon_min, lat_min, lon_max, lat_max = bbox
    for row in gdf.itertuples(index=False):
        seq_id = getattr(row, "sequence", None) or getattr(row, "sequence_id", None)
        image_id = getattr(row, "id", None)
        geom = getattr(row, "geometry", None)
        if not seq_id or not image_id or geom is None:
            continue
        if kept_ids and str(seq_id) not in kept_ids:
            continue
        lon = float(getattr(geom, "x"))
        lat = float(getattr(geom, "y"))
        if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
            continue
        width = _safe_int(getattr(row, "width", None))
        height = _safe_int(getattr(row, "height", None))
        cam_params = {}
        if width:
            cam_params["width"] = width
        if height:
            cam_params["height"] = height
        frame = FrameMeta(
            image_id=str(image_id),
            seq_id=str(seq_id),
            captured_at_ms=_captured_at_to_ms(getattr(row, "captured_at", None)),
            lon=lon,
            lat=lat,
            alt_ellip=_safe_float(getattr(row, "altitude", None)),
            camera_type=str(getattr(row, "camera_type", "unknown") or "unknown").lower(),
            cam_params=cam_params,
            quality_score=_safe_float(getattr(row, "quality_score", None)),
            thumbnail_url=getattr(row, "thumb_original_url", None),
        )
        sequences.setdefault(frame.seq_id, []).append(frame)

    for frames in sequences.values():
        frames.sort(key=lambda f: f.captured_at_ms)
    return sequences


def _dataset_kept_sequence_ids(dataset_dir: Path) -> set[str]:
    path = dataset_dir / "sequences" / "filtered_sequences.json"
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf8"))
    except json.JSONDecodeError:
        return set()
    details = payload.get("sequence_details")
    if isinstance(details, dict):
        return {str(key) for key in details.keys()}
    return set()


def _camera_pose_and_model_dicts(
    recon: dict[str, ReconstructionResult],
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    poses: dict[str, dict[str, object]] = {}
    models: dict[str, dict[str, object]] = {}
    for result in recon.values():
        for frame in result.frames:
            pose = result.poses.get(frame.image_id)
            if pose is None:
                continue
            R_cw = pose.R.T
            t_cw = -R_cw @ pose.t
            poses[frame.image_id] = {
                "rotation": R_cw.tolist(),
                "translation": t_cw.tolist(),
            }
            models[frame.image_id] = _camera_model_from_frame(frame)
    return poses, models


def _camera_model_from_frame(frame: FrameMeta) -> dict[str, object]:
    params = frame.cam_params or {}
    width = _safe_int(params.get("width") or params.get("image_width")) or 1920
    height = _safe_int(params.get("height") or params.get("image_height")) or 1080
    focal = _safe_float(params.get("focal"))
    if focal is None:
        focal_px = _safe_float(params.get("focal_px") or params.get("fx_px") or params.get("fx"))
        focal = float(focal_px / max(width, height)) if focal_px else 0.8
    return {"width": width, "height": height, "focal": float(focal)}


def _grid_transform_webmercator(
    grid: GridSpec,
    *,
    lon0: float,
    lat0: float,
) -> tuple[object, str]:
    from affine import Affine
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    mx0, my0 = transformer.transform(lon0, lat0)
    x_min = grid.ix_min * grid.res
    y_max = (grid.iy_min + grid.height) * grid.res
    transform = Affine.translation(mx0 + x_min, my0 + y_max) * Affine.scale(grid.res, -grid.res)
    return transform, "EPSG:3857"


def _points_and_attrs_from_consensus(points: list[dict[str, object]]) -> tuple[np.ndarray, dict[str, np.ndarray] | None]:
    if not points:
        return np.zeros((0, 3), dtype=np.float32), None
    xyz = np.asarray(
        [[float(p["x"]), float(p["y"]), float(p["z"])] for p in points],
        dtype=np.float32,
    )
    attrs = {
        "support": np.asarray([int(p.get("support", 0)) for p in points], dtype=np.int32),
        "sem_prob": np.asarray([float(p.get("sem_prob", np.nan)) for p in points], dtype=np.float32),
        "uncertainty": np.asarray([float(p.get("uncertainty", np.nan)) for p in points], dtype=np.float32),
    }
    return xyz, attrs


def _source_dtms_on_grid(
    sources: dict[str, list[GroundPoint]],
    grid: GridSpec,
) -> dict[str, np.ndarray]:
    results: dict[str, np.ndarray] = {}
    for name, points in sources.items():
        arr = np.full((grid.height, grid.width), np.nan, dtype=np.float32)
        buckets: dict[tuple[int, int], list[float]] = {}
        for point in points:
            ix = int(np.floor(point.x / grid.res)) - grid.ix_min
            iy = int(np.floor(point.y / grid.res)) - grid.iy_min
            if ix < 0 or iy < 0 or ix >= grid.width or iy >= grid.height:
                continue
            buckets.setdefault((iy, ix), []).append(float(point.z))
        for (iy, ix), heights in buckets.items():
            arr[iy, ix] = float(np.percentile(heights, constants.LOWER_ENVELOPE_Q * 100.0))
        results[name] = arr
    return results


def _summarize_reconstruction_sources(recon: dict[str, ReconstructionResult]) -> dict[str, object]:
    source_types: dict[str, int] = {}
    point_count = 0
    for result in recon.values():
        meta = result.metadata or {}
        source_type = str(meta.get("source_type") or ("fixture" if meta.get("fixture") else "unknown"))
        source_types[source_type] = source_types.get(source_type, 0) + 1
        point_count += int(result.points_xyz.shape[0])
    return {
        "sequence_count": len(recon),
        "source_types": source_types,
        "point_count": point_count,
    }


def _dataset_manifest(dataset_dir: Optional[Path]) -> dict[str, object] | None:
    if dataset_dir is None:
        return None
    config_path = dataset_dir / "config.json"
    if not config_path.exists():
        return {"path": str(dataset_dir)}
    try:
        config = json.loads(config_path.read_text(encoding="utf8"))
    except json.JSONDecodeError:
        return {"path": str(dataset_dir), "config_error": "invalid json"}
    return {"path": str(dataset_dir), "config": config}


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _token_hash(token: Optional[str]) -> str | None:
    if not token:
        return None
    return hashlib.sha256(token.encode("utf8")).hexdigest()[:16]


def _model_provenance_snapshot() -> dict[str, object]:
    manifest_path = Path(os.getenv("DTM_PRODUCTION_MODELS_MANIFEST", "models/production_models.json"))
    snapshot: dict[str, object] = {
        "manifest": str(manifest_path),
        "manifest_exists": manifest_path.exists(),
        "ground_mask_model_id": os.getenv("GROUND_MASK_MODEL_ID", "nvidia/segformer-b0-finetuned-cityscapes-512-1024"),
        "monodepth_model_id": os.getenv("MONODEPTH_MODEL_ID", "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"),
        "model_cache_dir": os.getenv("DTM_MODEL_CACHE_DIR", "models/huggingface"),
        "models_local_only": os.getenv("DTM_MODELS_LOCAL_ONLY", "1"),
    }
    if manifest_path.exists():
        try:
            snapshot["setup_manifest"] = json.loads(manifest_path.read_text(encoding="utf8"))
        except Exception as exc:
            snapshot["manifest_error"] = str(exc)
    return snapshot


def _captured_at_to_ms(raw) -> int:
    if raw is None:
        return 0
    if isinstance(raw, (int, float)):
        return int(raw)
    if isinstance(raw, str) and raw:
        from datetime import datetime, timezone

        try:
            raw_dt = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
            dt = datetime.fromisoformat(raw_dt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000.0)
        except ValueError:
            return 0
    return 0


def _safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


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


def _parse_float_list(raw: Optional[str]) -> list[float]:
    if not raw:
        return []
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf8")


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


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
