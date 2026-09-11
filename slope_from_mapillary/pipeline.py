"""Single production path from observed SfM tracks to slope products."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from . import __version__
from .estimation import FitPolicy
from .grid import Grid, estimate_cells, fuse_cells
from .output import sample_paths, write_json, write_report, write_surface
from .reconstruction import load, load_opensfm, load_colmap
from .reference import metric_crs, register_horizontal, vertical_basis
from .surfaces import ObservationPolicy, select_surfaces


def read_project(path: Path) -> dict:
    config = json.loads(Path(path).read_text())
    if config.get("schema_version") != 1 or not config.get("reconstructions"):
        raise ValueError("Project requires schema_version=1 and at least one reconstruction")
    surfaces = config.get("surfaces", {})
    for info in surfaces.values():
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", info["id"]):
            raise ValueError("Surface IDs must contain only letters, digits, underscores or hyphens")
        if info.get("kind") not in {"road", "sidewalk", "ramp", "terrain", "crossing"}:
            raise ValueError("Each surface needs a recognized physical kind")
    if not surfaces:
        raise ValueError("Project needs an explicit surface label mapping")
    ids = [spec["id"] for spec in config["reconstructions"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Reconstruction IDs must be unique")
    for spec in config["reconstructions"]:
        if not isinstance(spec.get("masks"), str) or not spec["masks"]:
            raise ValueError("Each reconstruction needs a surface mask directory")
        if not str(spec.get("evidence_group", "")).strip():
            raise ValueError("Each reconstruction needs an evidence_group (shared images = same group)")
        allowance = spec.get("geometry_uncertainty_deg")
        if allowance is not None and (not np.isfinite(allowance) or not 0 <= allowance < 45 or
                                      not str(spec.get("geometry_uncertainty_evidence", "")).strip()):
            raise ValueError("Geometry uncertainty needs a finite angular bound and calibration evidence")
    ObservationPolicy(**config.get("observation_policy", {}))
    FitPolicy(**config.get("fit_policy", {}))
    return config


def _horizontal_controls(spec: dict, crs) -> dict:
    from pyproj import Transformer

    value = dict(spec)
    project = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    controls = []
    for row in value.get("controls", []):
        row = dict(row)
        if "longitude" in row or "latitude" in row:
            lon, lat = float(row["longitude"]), float(row["latitude"])
            if not np.isfinite([lon, lat]).all() or not (-180 <= lon <= 180 and -90 <= lat <= 90):
                raise ValueError("Invalid horizontal GNSS control")
            row["x"], row["y"] = project.transform(lon, lat)
        controls.append(row)
    value["controls"] = controls
    return value


def audit_project(path: Path) -> dict:
    """Read actual inputs and identify measurement blockers without inventing data."""
    path = Path(path).resolve()
    config, root = read_project(path), path.parent
    results, shared_images = [], {}
    for spec in config["reconstructions"]:
        result = {"id": spec["id"], "blockers": [], "notes": []}
        try:
            reconstruction = load(spec, root)
            result.update(registered_images=len(reconstruction.shots), reconstructed_points=len(reconstruction.points),
                          points_with_tracks=sum(bool(p.observations) for p in reconstruction.points.values()))
            for name in reconstruction.shots:
                if name in shared_images and shared_images[name] != spec["evidence_group"]:
                    result["blockers"].append("Shared images assigned to different evidence groups")
                    break
                shared_images[name] = spec["evidence_group"]
            try:
                vertical_basis(spec.get("vertical_reference"))
            except (ValueError, TypeError) as exc:
                result["blockers"].append(str(exc))
            if not spec.get("georeference", {}).get("controls"):
                result["blockers"].append("Horizontal camera controls are missing")
            cloud = select_surfaces(reconstruction, root / spec.get("masks", "masks"),
                                    config["surfaces"], ObservationPolicy(**config.get("observation_policy", {})))
            result["surface_selection"] = cloud.stats
            if not len(cloud.xyz):
                result["blockers"].append("No surface points passed observed multi-view semantic/geometry gates")
            if spec.get("geometry_uncertainty_deg") is None:
                result["notes"].append("Geometry accuracy uncalibrated; threshold screening will be uncertainty_unknown")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result["blockers"].append(str(exc))
        results.append(result)
    return {"status": "blocked" if any(row["blockers"] for row in results) else "inputs_ready",
            "reconstructions": results,
            "scope": "Input evidence audit; CRS/registration and local support are checked during run"}


def _hash(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_pipeline(project_path: Path, out_dir: Path) -> dict:
    project_path, out_dir = Path(project_path).resolve(), Path(out_dir).resolve()
    config, root = read_project(project_path), project_path.parent
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("Output directory must be new or empty, to prevent mixing products from different runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_project(project_path)
    write_json(out_dir / "audit.json", audit)
    if audit["status"] == "blocked":
        raise ValueError("Required measurement evidence is missing; see audit.json")
    crs = metric_crs(config["crs"])
    policy = FitPolicy(**config.get("fit_policy", {}))
    observation_policy = ObservationPolicy(**config.get("observation_policy", {}))
    prepared, evidence = [], []
    inputs = {str(project_path): _hash(project_path)}
    for spec in config["reconstructions"]:
        reconstruction = load(spec, root)
        reference = spec["vertical_reference"]
        registration = register_horizontal(reconstruction, vertical_basis(reference),
                                           _horizontal_controls(spec["georeference"], crs))
        cloud = select_surfaces(reconstruction, root / spec["masks"], config["surfaces"], observation_policy)
        xyz = registration.transform(cloud.xyz)
        prepared.append((spec, cloud, xyz, registration))
        evidence.append({"id": spec["id"], "evidence_group": spec["evidence_group"],
                         "source": reconstruction.source, "surface_selection": cloud.stats,
                         "vertical_reference": reference,
                         "geometry_uncertainty_deg": spec.get("geometry_uncertainty_deg"),
                         "geometry_uncertainty_evidence": spec.get("geometry_uncertainty_evidence"),
                         "horizontal_registration_rmse_m": registration.rmse_m,
                         "horizontal_registration_scale": registration.scale,
                         "rejected_controls": registration.rejected_controls})
        source = root / spec["path"]
        source_paths = [source] if source.is_file() else [source / name for name in
                                                       ("cameras.txt", "images.txt", "points3D.txt")]
        if spec.get("tracks"):
            source_paths.append(root / spec["tracks"])
        source_paths.extend((root / spec["masks"]).glob("*.npz"))
        inputs.update({str(path.resolve()): _hash(path) for path in source_paths})
    grid = Grid.from_points(np.concatenate([item[2] for item in prepared]),
                            float(config.get("resolution_m", 0.5)), int(config.get("max_cells", 2_000_000)))
    surface_results, rejected, conflicts = {}, {}, {}
    for surface in config["surfaces"].values():
        surface_id = surface["id"]
        candidates = defaultdict(list)
        rejected[surface_id] = {}
        for spec, cloud, xyz, registration in prepared:
            indices = [i for i, name in enumerate(cloud.surface_ids) if name == surface_id]
            if not indices:
                continue
            cells, reasons = estimate_cells(xyz[indices], [cloud.point_ids[i] for i in indices], grid, policy,
                source_id=spec["id"], evidence_group=spec["evidence_group"],
                vertical_uncertainty=spec["vertical_reference"].get("uncertainty_deg"),
                geometry_uncertainty=spec.get("geometry_uncertainty_deg"), registration_rmse_m=registration.rmse_m)
            rejected[surface_id][spec["id"]] = reasons
            for key, measurement in cells.items():
                candidates[key].append(measurement)
        surface_results[surface_id], conflicts[surface_id] = fuse_cells(
            candidates, float(config.get("max_disagreement_deg", 2.0)))
    manifest = {"schema_version": 1, "software_version": __version__,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "estimated" if any(surface_results.values()) else "no_supported_cells",
                "validation_status": "not_field_validated", "crs": crs.to_string(),
                "grid": asdict(grid), "fit_policy": asdict(policy),
                "observation_policy": asdict(observation_policy), "evidence": evidence,
                "accepted_cells": {key: len(value) for key, value in surface_results.items()},
                "measured_area_m2": {key: len(value)*grid.resolution**2 for key, value in surface_results.items()},
                "rejection_counts": rejected, "conflicting_cells": conflicts,
                "input_sha256": inputs,
                "uncertainty_note": "Fit precision excludes SfM correlations and bias. Angular allowance is a screening budget, not a confidence interval.",
                "spatial_note": "Cell resolution is not positional accuracy. Inspect registration RMSE before matching narrow paths.",
                "attribution": config.get("attribution", "Mapillary imagery contributors; retain source attribution")}
    # Persist the complete numerical outcome before optional exports. Exceptions
    # mark an incomplete run, never leave an apparently successful manifest.
    manifest["export_complete"] = False
    write_json(out_dir / "manifest.json", manifest)
    for surface_id, cells in surface_results.items():
        write_surface(out_dir / surface_id, surface_id, cells, grid, crs,
                      lineage=bool(config.get("write_lineage", True)))
    if config.get("paths"):
        paths_file = root / config["paths"]
        thresholds = config.get("screening_thresholds", {})
        for name in ("longitudinal_pct", "cross_slope_pct"):
            if name not in thresholds or not np.isfinite(thresholds[name]) or thresholds[name] < 0:
                raise ValueError("Path screening requires explicit nonnegative longitudinal/cross_slope_pct thresholds")
        path_output, summaries = sample_paths(json.loads(paths_file.read_text()), surface_results, grid, crs, thresholds)
        write_json(out_dir / "path_slopes.geojson", path_output)
        manifest["path_coverage"] = summaries
        manifest["screening_thresholds"] = thresholds
        manifest["input_sha256"][str(paths_file)] = _hash(paths_file)
    manifest["export_complete"] = True
    write_json(out_dir / "manifest.json", manifest)
    write_report(out_dir / "report.html", manifest)
    return manifest


def audit_dataset(dataset: Path) -> dict:
    """Inspect the authoritative real sample without treating it as calibration."""
    dataset = Path(dataset)
    reconstructions = []
    for path in sorted(dataset.glob("geometry/opensfm/**/reconstruction.json")):
        payload = json.loads(path.read_text())
        for component in range(len(payload)):
            tracks = path.with_name("tracks.csv")
            model = load_opensfm(path, tracks if tracks.exists() else None, component)
            reconstructions.append({"format": "opensfm", "path": str(path), "component": component,
                                    "images": len(model.shots), "points": len(model.points),
                                    "tracked_points": sum(bool(p.observations) for p in model.points.values())})
    for path in sorted(dataset.glob("geometry/colmap/**/points3D.txt")):
        model = load_colmap(path.parent)
        reconstructions.append({"format": "colmap", "path": str(path.parent),
                                "images": len(model.shots), "points": len(model.points),
                                "tracked_points": sum(bool(p.observations) for p in model.points.values())})
    return {"status": "needs_slope_evidence", "reconstructions": reconstructions,
            "required": ["Independent calibrated vertical reference for each reconstruction",
                         "Observed road/sidewalk/level-specific semantic labels on the reconstructed images",
                         "Horizontal registration controls and a local metric CRS",
                         "Held-out field measurements of longitudinal and cross slope"],
            "note": "Existing depth/height products, EXIF orientation and reconstruction Z are not substitutes for this evidence."}
