"""Command line for the slope-first project."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from .output import write_json
from .pipeline import audit_dataset, audit_project, run_pipeline
from .reconstruction import load


def prepare_project(reconstruction: Path, format: str, output: Path,
                    *, tracks: Path | None = None, metadata: Path | None = None,
                    component: int = 0) -> dict:
    """Create an honest project template from real inputs; missing evidence is null."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Project already exists: {output}")
    relative = lambda path: os.path.relpath(Path(path).resolve(), output.parent)
    spec = {"id": "capture-1", "evidence_group": "capture-1", "format": format,
            "path": relative(reconstruction), "component": component,
            "masks": "surface_masks", "vertical_reference": None,
            "geometry_uncertainty_deg": None, "geometry_uncertainty_evidence": None,
            "georeference": {"source": "Mapillary horizontal GNSS metadata", "controls": [],
                             "max_residual_m": 3.0, "min_baseline_m": 10.0}}
    if tracks is not None:
        spec["tracks"] = relative(tracks)
    model = load(spec, output.parent)
    crs = None
    if metadata is not None:
        payload = json.loads(Path(metadata).read_text())
        rows = [row for values in payload.values() for row in values] if isinstance(payload, dict) else payload
        by_id = {str(row["image_id"]): row for row in rows}
        for name in model.shots:
            stem = Path(name).stem
            row = by_id.get(stem) or by_id.get(stem.split("_")[-1])
            if row is None:
                continue
            spec["georeference"]["controls"].append({"image": name, "longitude": row["lon"],
                                                        "latitude": row["lat"]})
        controls = spec["georeference"]["controls"]
        if controls:
            lon = float(np.median([row["longitude"] for row in controls]))
            lat = float(np.median([row["latitude"] for row in controls]))
            if -80 <= lat <= 84:
                zone = min(60, max(1, int((lon + 180) // 6) + 1))
                crs = f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"
    config = {"schema_version": 1, "crs": crs, "resolution_m": 0.5, "max_cells": 2_000_000,
              "surfaces": {"1": {"id": "road", "kind": "road"},
                           "2": {"id": "sidewalk", "kind": "sidewalk"},
                           "3": {"id": "terrain", "kind": "terrain"}},
              "reconstructions": [spec], "paths": None,
              "screening_thresholds": None, "max_disagreement_deg": 2.0, "write_lineage": True,
              "attribution": "Mapillary imagery contributors; see source image IDs in reconstruction/lineage"}
    write_json(output, config)
    return {"project": str(output), "registered_images": len(model.shots),
            "reconstructed_points": len(model.points),
            "next": "Supply an independent vertical reference, prepare/review class masks, then audit and run. See documentation/METHOD.md."}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Observed surface slope models for accessibility")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="Check a slope project's actual input evidence")
    audit.add_argument("project", type=Path)
    sample = commands.add_parser("audit-dataset", help="Inspect a real cached dataset without inventing calibration")
    sample.add_argument("dataset", type=Path)
    sample.add_argument("--out", type=Path)
    prepare = commands.add_parser("prepare", help="Create a slope project template from a reconstruction")
    prepare.add_argument("reconstruction", type=Path)
    prepare.add_argument("--format", required=True, choices=["opensfm", "colmap"])
    prepare.add_argument("--tracks", type=Path)
    prepare.add_argument("--component", type=int, default=0)
    prepare.add_argument("--metadata", type=Path)
    prepare.add_argument("--out", required=True, type=Path)
    run = commands.add_parser("run", help="Estimate observed surface slopes and export maps")
    run.add_argument("project", type=Path)
    run.add_argument("--out-dir", required=True, type=Path)
    segment = commands.add_parser("segment", help="Run real semantic segmentation on registered images")
    segment.add_argument("project", type=Path)
    segment.add_argument("--images", type=Path, required=True)
    segment.add_argument("--model", default="nvidia/segformer-b0-finetuned-cityscapes-512-1024")
    segment.add_argument("--device", default="cpu")
    segment.add_argument("--local-files-only", action="store_true")
    validate = commands.add_parser("validate", help="Compare exports with independent field slope observations")
    validate.add_argument("model_dir", type=Path)
    validate.add_argument("--reference", required=True, type=Path)
    validate.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_project(args.reconstruction, args.format, args.out, tracks=args.tracks,
                                     metadata=args.metadata, component=args.component)
        elif args.command == "audit":
            result = audit_project(args.project)
        elif args.command == "audit-dataset":
            result = audit_dataset(args.dataset)
            if args.out:
                write_json(args.out, result)
        elif args.command == "segment":
            from .segmentation import segment_project
            result = segment_project(args.project, args.images, model_id=args.model, device=args.device,
                                     local_files_only=args.local_files_only)
        elif args.command == "validate":
            from .validation import validate_model
            result = validate_model(args.model_dir, args.reference, args.out)
        else:
            result = run_pipeline(args.project, args.out_dir)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 2 if result.get("status") in {"blocked", "no_supported_cells", "no_overlap"} else 0
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(f"Slope model could not be produced: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
