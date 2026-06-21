"""
GeoJSON writers for frame metadata and camera positions.

Two outputs are produced by the pipeline:

``frames.geojson``
    Written immediately after the ingestion stage.  Contains one Feature per
    downloaded frame with a Point geometry at the raw GNSS camera position
    (lon, lat, alt_ellip).  This file is always present and can be opened
    directly in QGIS, geojson.io, or any GIS tool to verify coverage before
    reconstruction begins.

``camera_positions.geojson``
    Written after the fusion/writers stage when at least one reconstruction
    (OpenSfM, COLMAP, or VO) has succeeded.  Contains one Feature per camera
    for which a pose was recovered, with the position converted from the local
    ENU frame back to WGS84.  A ``source`` property records which reconstruction
    provided the pose so per-source coverage can be compared.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_frames_geojson(
    seqs: Dict[str, List],
    out_path: Path | str,
    *,
    indent: int = 2,
) -> str:
    """Write a GeoJSON FeatureCollection of raw GNSS camera positions.

    Parameters
    ----------
    seqs:
        Mapping of ``sequence_id -> List[FrameMeta]`` as returned by the
        ingestion stage.
    out_path:
        Destination file path.  Parent directories are created automatically.
    indent:
        JSON indentation level (default 2).  Pass ``None`` for compact output.

    Returns
    -------
    str
        Absolute path to the written file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    features: List[dict] = []
    for frames in seqs.values():
        for frame in frames:
            try:
                features.append(frame.to_geojson_feature())
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("Skipping frame %s in GeoJSON export: %s", getattr(frame, "image_id", "?"), exc)

    collection = _feature_collection(features)
    _write_geojson(out_path, collection, indent=indent)
    log.info("Wrote %d frame positions to %s", len(features), out_path)
    return str(out_path)


def write_camera_positions_geojson(
    recon: Dict[str, object],
    lon0: float,
    lat0: float,
    h0: float,
    out_path: Path | str,
    *,
    source_label: str = "reconstruction",
    indent: int = 2,
) -> str:
    """Write a GeoJSON FeatureCollection of SfM-refined camera positions.

    Camera poses stored in a ``ReconstructionResult`` are expressed in a local
    ENU frame centred on ``(lon0, lat0, h0)``.  This function converts each
    camera centre back to WGS84 so it can be visualised alongside the raw GNSS
    positions from ``write_frames_geojson``.

    Parameters
    ----------
    recon:
        Mapping of ``sequence_id -> ReconstructionResult`` as returned by
        ``run_opensfm``, ``run_colmap``, or ``run_vo``.
    lon0, lat0, h0:
        ENU origin in WGS84 (degrees, degrees, metres ellipsoidal).
    out_path:
        Destination file path.
    source_label:
        Value placed in the ``source`` property of each Feature so the caller
        can distinguish opensfm / colmap / vo outputs when overlaying all three.
    indent:
        JSON indentation level.

    Returns
    -------
    str
        Absolute path to the written file.
    """
    from ..common_core import enu_to_wgs84

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    features: List[dict] = []
    for result in recon.values():
        for frame in result.frames:
            pose = result.poses.get(frame.image_id)
            if pose is None:
                continue
            # Camera centre in world frame (ENU): C = -R^T @ t  (world-from-camera)
            import numpy as np
            centre_enu = -pose.R.T @ pose.t
            try:
                lon, lat, h = enu_to_wgs84(
                    float(centre_enu[0]),
                    float(centre_enu[1]),
                    float(centre_enu[2]),
                    lon0, lat0, h0,
                )
            except Exception as exc:  # pragma: no cover - pyproj failure
                log.debug("ENU->WGS84 failed for %s: %s", frame.image_id, exc)
                continue

            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat, h],
                },
                "properties": {
                    "image_id": frame.image_id,
                    "seq_id": frame.seq_id,
                    "captured_at_ms": int(frame.captured_at_ms),
                    "camera_type": frame.camera_type,
                    "source": source_label,
                    # Raw GNSS position for cross-reference
                    "gnss_lon": float(frame.lon),
                    "gnss_lat": float(frame.lat),
                    "gnss_alt_ellip": float(frame.alt_ellip) if frame.alt_ellip is not None else None,
                },
            })

    collection = _feature_collection(features)
    _write_geojson(out_path, collection, indent=indent)
    log.info(
        "Wrote %d SfM camera positions (%s) to %s",
        len(features), source_label, out_path,
    )
    return str(out_path)


def write_all_camera_positions_geojson(
    reconstructions: Dict[str, Dict[str, object]],
    lon0: float,
    lat0: float,
    h0: float,
    scales: Dict[str, float],
    out_dir: Path | str,
    *,
    indent: int = 2,
) -> List[str]:
    """Write GeoJSON files combining calibrated camera positions.

    Parameters
    ----------
    reconstructions:
        Mapping of ``source_label -> {seq_id: ReconstructionResult}``.
        For example ``{"opensfm": reconA, "colmap": reconB, "vo": vo}``.
    lon0, lat0, h0:
        ENU origin in WGS84.
    scales:
        Mapping of ``seq_id -> float`` from the height solver to scale translations.
    out_dir:
        Destination directory. Outputs will be named ``camera_positions_{source_label}.geojson``.

    Returns
    -------
    List[str]
        Absolute paths to the written files.
    """
    from ..common_core import enu_to_wgs84
    import numpy as np

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths_written: List[str] = []

    for source_label, recon in reconstructions.items():
        if not recon:
            continue
            
        features: List[dict] = []
        for result in recon.values():
            scale = float(scales.get(result.seq_id, 1.0))
            for frame in result.frames:
                pose = result.poses.get(frame.image_id)
                if pose is None:
                    continue
                
                # In our pipeline, pose.t already represents the camera center in ENU.
                centre_enu_scaled = np.asarray(pose.t, dtype=float) * scale
                
                try:
                    lon, lat, h = enu_to_wgs84(
                        float(centre_enu_scaled[0]),
                        float(centre_enu_scaled[1]),
                        float(centre_enu_scaled[2]),
                        lon0, lat0, h0,
                    )
                except Exception as exc:
                    log.debug("ENU->WGS84 failed for %s/%s: %s", source_label, frame.image_id, exc)
                    continue

                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat, h],
                    },
                    "properties": {
                        "image_id": frame.image_id,
                        "seq_id": frame.seq_id,
                        "captured_at_ms": int(frame.captured_at_ms),
                        "camera_type": frame.camera_type,
                        "source": source_label,
                        "applied_scale": scale,
                        "gnss_lon": float(frame.lon),
                        "gnss_lat": float(frame.lat),
                        "gnss_alt_ellip": float(frame.alt_ellip) if frame.alt_ellip is not None else None,
                    },
                })
        
        if features:
            collection = _feature_collection(features)
            out_path = out_dir / f"camera_positions_{source_label}.geojson"
            _write_geojson(out_path, collection, indent=indent)
            log.info("Wrote %d calibrated SfM camera positions to %s", len(features), out_path)
            paths_written.append(str(out_path))

    return paths_written


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _feature_collection(features: List[dict]) -> dict:
    return {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": features,
    }


def _write_geojson(path: Path, collection: dict, *, indent: Optional[int]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(
            json.dumps(collection, indent=indent, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(path)
    except Exception as exc:  # pragma: no cover - I/O error
        log.error("Failed to write GeoJSON to %s: %s", path, exc)
        raise
