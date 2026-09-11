"""Slope products and directional path samples; unknown never means flat."""
from __future__ import annotations

from html import escape
import json
from pathlib import Path

import numpy as np

from .estimation import directional_allowance, directional_grades, screening_status
from .grid import Grid


def write_json(path: Path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def cell_properties(cell):
    magnitude = float(np.linalg.norm(cell.gradient))
    return {
        "slope_pct": 100 * magnitude,
        "slope_deg": float(np.degrees(np.arctan(magnitude))),
        "gradient_east": float(cell.gradient[0]), "gradient_north": float(cell.gradient[1]),
        "downslope_aspect_deg": (None if magnitude < 1e-12 else
                                 float(np.degrees(np.arctan2(-cell.gradient[0], -cell.gradient[1])) % 360)),
        "angular_allowance_deg": cell.bound_deg,
        "fit_precision_deg": cell.fit_precision_deg,
        "independent_capture_groups": cell.independent_groups,
        "sources": cell.sources,
        "point_count": sum(obs.plane.point_count for obs in cell.observations),
        "registration_rmse_m": max(obs.registration_rmse_m for obs in cell.observations),
        "validation_status": "not_field_validated",
    }


def write_surface(out: Path, surface_id: str, cells: dict, grid: Grid, crs, *, lineage: bool):
    import rasterio
    from rasterio.transform import from_origin
    from pyproj import Transformer

    out.mkdir(parents=True, exist_ok=True)
    projector = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    fields = ["slope_pct", "slope_deg", "gradient_east", "gradient_north",
              "downslope_aspect_deg", "angular_allowance_deg", "fit_precision_deg",
              "independent_capture_groups", "point_count", "registration_rmse_m"]
    arrays = {key: np.full((grid.height, grid.width), np.nan, dtype=np.float32) for key in fields}
    features, evidence = [], []
    for (row, col), cell in sorted(cells.items()):
        props = cell_properties(cell)
        props.update(surface_id=surface_id, row=row, col=col)
        for field in fields:
            if props[field] is not None:
                arrays[field][row, col] = props[field]
        center = grid.center(row, col)
        corners = center + np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]) * grid.resolution / 2
        lon, lat = projector.transform(corners[:, 0], corners[:, 1])
        features.append({"type": "Feature", "geometry": {"type": "Polygon",
                         "coordinates": [np.column_stack([lon, lat]).tolist()]}, "properties": props})
        if lineage:
            evidence.append({"row": row, "col": col, "observations": [
                {"source_id": obs.source_id, "candidate_point_ids": obs.point_ids}
                for obs in cell.observations]})
    transform = from_origin(grid.west, grid.north, grid.resolution, grid.resolution)
    for field, array in arrays.items():
        with rasterio.open(out / f"{field}.tif", "w", driver="GTiff", width=grid.width,
                           height=grid.height, count=1, dtype="float32", crs=crs,
                           transform=transform, nodata=np.nan, compress="deflate") as dst:
            dst.write(array, 1)
            dst.set_band_description(1, field)
            dst.update_tags(surface_id=surface_id, quantity=field,
                            validation_status="not_field_validated", no_extrapolation="true")
    write_json(out / "patches.geojson", {"type": "FeatureCollection", "features": features})
    if lineage:
        write_json(out / "lineage.json", evidence)


def sample_paths(path_data: dict, surfaces: dict, grid: Grid, crs, thresholds: dict):
    """Split paths at EVERY raster boundary, preserving unknown lengths.

    Input is WGS84 GeoJSON LineString/MultiLineString. Each feature explicitly
    names its surface_id; road results are never transferred to a sidewalk.
    Grade is relative to coordinate order, and cross grade rises to its right.
    """
    from pyproj import Transformer

    if path_data.get("type") != "FeatureCollection":
        raise ValueError("Paths must be a WGS84 GeoJSON FeatureCollection")
    project = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    unproject = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    features, summaries = [], []
    for index, feature in enumerate(path_data["features"]):
        props = feature.get("properties") or {}
        surface_id = props.get("surface_id")
        if surface_id not in surfaces:
            raise ValueError(f"Path {index} needs an explicit known surface_id")
        geometry = feature["geometry"]
        if geometry["type"] not in {"LineString", "MultiLineString"}:
            raise ValueError("Path geometry must be LineString or MultiLineString")
        parts = [geometry["coordinates"]] if geometry["type"] == "LineString" else geometry["coordinates"]
        length_total, length_measured = 0.0, 0.0
        for part in parts:
            coords = np.asarray(part, dtype=float)
            if coords.ndim != 2 or coords.shape[0] < 2 or coords.shape[1] < 2:
                raise ValueError("Path part must have at least two geographic coordinates")
            if not np.isfinite(coords).all() or np.any(np.abs(coords[:, 0]) > 180) or np.any(np.abs(coords[:, 1]) > 90):
                raise ValueError("Invalid WGS84 path coordinates")
            xy = np.column_stack(project.transform(coords[:, 0], coords[:, 1]))
            for start, end in zip(xy[:-1], xy[1:]):
                tangent = end - start
                length = float(np.linalg.norm(tangent))
                if length < 1e-9:
                    continue
                # Analytical intersection parameters with the finite grid's
                # horizontal and vertical boundaries. Outside spans stay unknown.
                cuts = [0.0, 1.0]
                for axis, base, count, step in ((0, grid.west, grid.width, grid.resolution),
                                                (1, grid.north, grid.height, -grid.resolution)):
                    if abs(tangent[axis]) < 1e-12:
                        continue
                    ends = np.sort((np.array([start[axis], end[axis]]) - base) / step)
                    first, last = max(0, int(np.ceil(ends[0]))), min(count, int(np.floor(ends[1])))
                    t = (base + np.arange(first, last + 1) * step - start[axis]) / tangent[axis]
                    cuts.extend(t[(t > 0) & (t < 1)].tolist())
                cuts = np.unique(cuts)
                for lo, hi in zip(cuts[:-1], cuts[1:]):
                    a, b = start + lo * tangent, start + hi * tangent
                    segment_length = (hi-lo) * length
                    if segment_length < 1e-9:
                        continue
                    length_total += segment_length
                    key = grid.index((a+b)/2)
                    cell = surfaces[surface_id].get(key)
                    details = {"path_id": str(feature.get("id", props.get("id", index))),
                               "surface_id": surface_id, "length_m": segment_length,
                               "longitudinal_pct": None, "cross_slope_pct": None,
                               "angular_allowance_deg": None,
                               "longitudinal_screening": "unknown", "cross_slope_screening": "unknown",
                               "validation_status": "not_field_validated"}
                    if cell is not None:
                        along, across = directional_grades(cell.gradient, tangent)
                        along_bound = directional_allowance(cell.gradient, tangent, cell.bound_deg)
                        across_bound = directional_allowance(cell.gradient, np.array([tangent[1], -tangent[0]]), cell.bound_deg)
                        details.update(longitudinal_pct=along, cross_slope_pct=across,
                                       angular_allowance_deg=cell.bound_deg,
                                       longitudinal_allowance_deg=along_bound,
                                       cross_slope_allowance_deg=across_bound,
                                       longitudinal_screening=screening_status(along, along_bound,
                                                                              thresholds["longitudinal_pct"]),
                                       cross_slope_screening=screening_status(across, across_bound,
                                                                             thresholds["cross_slope_pct"]))
                        length_measured += segment_length
                    lon, lat = unproject.transform([a[0], b[0]], [a[1], b[1]])
                    features.append({"type": "Feature", "geometry": {"type": "LineString",
                                     "coordinates": np.column_stack([lon, lat]).tolist()},
                                     "properties": details})
        summaries.append({"path_index": index, "length_m": length_total,
                          "measured_length_m": length_measured,
                          "unknown_length_m": max(0.0, length_total-length_measured),
                          "coverage_fraction": length_measured / length_total if length_total else 0.0})
    return {"type": "FeatureCollection", "features": features}, summaries


def write_report(path: Path, manifest: dict):
    summary = escape(json.dumps(manifest, indent=2, allow_nan=False))
    path.write_text("<!doctype html><html lang='en'><meta charset='utf-8'>"
                    "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                    "<title>Mapillary slope model report</title><style>"
                    "body{font:16px system-ui;max-width:1000px;margin:40px auto;padding:0 20px;line-height:1.5}"
                    "pre{background:#f3f5f6;padding:20px;overflow:auto}"
                    "</style><h1>Observed surface slope model</h1>"
                    "<p>These estimates have not been validated against field slope measurements. "
                    "Fit precision is separate from measurement accuracy. Unknown cells and path "
                    "segments have no usable estimate. Screening thresholds are project settings.</p>"
                    "<h2>Coverage, evidence and rejection reasons</h2><pre>" + summary + "</pre></html>")
