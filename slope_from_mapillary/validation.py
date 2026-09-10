"""Evaluate exported gradients against independent field slope observations."""
from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np

from .estimation import directional_grades
from .output import write_json


def validate_model(model_dir: Path, reference: Path, out: Path) -> dict:
    """Read held-out WGS84 points with grid bearings and measured signed grades.

    References never enter reconstruction, registration, fitting or fusion.
    This evaluation does not automatically certify a whole map or change its
    validation_status: representativeness and instrument quality need review.
    """
    import rasterio
    from pyproj import Transformer

    model_dir = Path(model_dir)
    manifest = json.loads((model_dir / "manifest.json").read_text())
    if not manifest.get("export_complete"):
        raise ValueError("Cannot validate an incomplete export")
    payload = json.loads(Path(reference).read_text())
    if payload.get("type") != "FeatureCollection" or not payload.get("features"):
        raise ValueError("Reference must be a nonempty WGS84 GeoJSON FeatureCollection")
    project = Transformer.from_crs("EPSG:4326", manifest["crs"], always_xy=True)
    rows, known_ids = [], set()
    with ExitStack() as stack:
        datasets = {}
        for feature in payload["features"]:
            props = feature["properties"]
            identifier = str(props["reference_id"])
            if identifier in known_ids:
                raise ValueError("Field reference IDs must be unique")
            known_ids.add(identifier)
            surface = props["surface_id"]
            if surface not in manifest["accepted_cells"]:
                raise ValueError("Reference names an unknown surface_id")
            if not props.get("instrument") or not props.get("measured_at"):
                raise ValueError("Reference requires instrument and measurement date provenance")
            if feature["geometry"]["type"] != "Point":
                raise ValueError("Field reference geometry must be a WGS84 Point")
            coordinates = feature["geometry"]["coordinates"]
            lon, lat = coordinates[:2]
            if not np.isfinite([lon, lat]).all() or abs(lon) > 180 or abs(lat) > 90:
                raise ValueError("Invalid reference WGS84 coordinates")
            bearing = float(props["bearing_grid_deg"])
            observed = np.array([props["longitudinal_pct"], props["cross_slope_pct"]], dtype=float)
            if not np.isfinite(observed).all() or not np.isfinite(bearing):
                raise ValueError("Reference slope and grid bearing must be finite")
            if surface not in datasets:
                datasets[surface] = [stack.enter_context(rasterio.open(model_dir / surface / f"gradient_{axis}.tif"))
                                     for axis in ("east", "north")]
            x, y = project.transform(lon, lat)
            east, north = datasets[surface]
            row, col = east.index(x, y)
            result = {"reference_id": identifier, "surface_id": surface, "status": "unknown",
                      "instrument": props["instrument"], "measured_at": props["measured_at"]}
            if 0 <= row < east.height and 0 <= col < east.width:
                gradient = np.array([next(src.sample([(x, y)]))[0] for src in (east, north)])
                if np.isfinite(gradient).all():
                    angle = np.radians(bearing)
                    predicted = np.array(directional_grades(gradient, np.array([np.sin(angle), np.cos(angle)])))
                    result.update(status="compared", observed_pct=observed.tolist(),
                                  predicted_pct=predicted.tolist(), error_percentage_points=(predicted-observed).tolist())
            rows.append(result)
    groups = {}
    for surface in sorted({row["surface_id"] for row in rows}):
        selected = [row for row in rows if row["surface_id"] == surface]
        compared = [row for row in selected if row["status"] == "compared"]
        group = {"reference_count": len(selected), "compared_count": len(compared),
                 "unknown_count": len(selected)-len(compared)}
        if compared:
            errors = np.array([row["error_percentage_points"] for row in compared])
            group["metrics_percentage_points"] = {
                name: {"bias": float(errors[:, i].mean()),
                       "mae": float(np.abs(errors[:, i]).mean()),
                       "rmse": float(np.sqrt(np.mean(errors[:, i]**2))),
                       "absolute_error_p95": float(np.percentile(np.abs(errors[:, i]), 95))}
                for i, name in enumerate(("longitudinal", "cross_slope"))}
        groups[surface] = group
    result = {"status": "compared" if any(row["status"] == "compared" for row in rows) else "no_overlap",
              "by_surface": groups, "observations": rows,
              "note": "Review reference accuracy, horizontal registration and sampling coverage before interpreting these errors."}
    write_json(out, result)
    return result
