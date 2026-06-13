from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dtm_from_mapillary.cli.pipeline import (
    _camera_pose_and_model_dicts,
    _grid_transform_webmercator,
    run_pipeline,
)
from dtm_from_mapillary.common_core import FrameMeta, Pose, ReconstructionResult
from dtm_from_mapillary.fusion.heightmap_fusion import GridSpec, fuse
from dtm_from_mapillary.geom.opensfm_adapter import OpenSfMUnavailable
from dtm_from_mapillary.geom.sfm_opensfm import run as run_opensfm
from scripts.check_sample_dataset import validate_sample_dataset


def _frame(image_id: str = "img-1") -> FrameMeta:
    return FrameMeta(
        image_id=image_id,
        seq_id="seq-1",
        captured_at_ms=0,
        lon=-48.0,
        lat=-27.0,
        alt_ellip=10.0,
        camera_type="perspective",
        cam_params={"width": 1920, "height": 1080, "focal": 0.8},
        quality_score=0.9,
    )


def test_fusion_preserves_unsupported_cells_as_nan() -> None:
    points = [
        {"x": 0.0, "y": 0.0, "z": 10.0, "sources": ["A", "B"]},
        {"x": 2.0, "y": 0.0, "z": 12.0, "sources": ["A", "B"]},
    ]
    dtm, conf, grid = fuse(points, grid_res=1.0, return_grid=True)
    assert isinstance(grid, GridSpec)
    assert np.isfinite(dtm[0, 0])
    assert np.isnan(dtm[0, 1])
    assert np.isfinite(dtm[0, 2])
    assert conf[0, 1] == 0.0


def test_breakline_pose_extraction_uses_reconstruction_results() -> None:
    frame = _frame()
    pose = Pose(R=np.eye(3), t=np.array([1.0, 2.0, 3.0]))
    recon = ReconstructionResult(
        seq_id="seq-1",
        frames=[frame],
        poses={frame.image_id: pose},
        points_xyz=np.zeros((0, 3), dtype=np.float32),
        source="opensfm",
    )
    poses, models = _camera_pose_and_model_dicts({"seq-1": recon})
    assert poses[frame.image_id]["translation"] == [-1.0, -2.0, -3.0]
    assert models[frame.image_id]["width"] == 1920
    assert models[frame.image_id]["focal"] == 0.8


def test_grid_transform_is_georeferenced() -> None:
    transform, crs = _grid_transform_webmercator(
        GridSpec(ix_min=-2, iy_min=3, width=4, height=5, res=0.5),
        lon0=-48.0,
        lat0=-27.0,
    )
    assert crs == "EPSG:3857"
    assert transform.a == 0.5
    assert transform.e == -0.5


def test_opensfm_strict_mode_does_not_fallback_to_synthetic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPEN_SFM_FORCE_SYNTHETIC", "1")
    with pytest.raises(OpenSfMUnavailable):
        run_opensfm({"seq-1": [_frame()]}, allow_synthetic=False)


def test_sample_dataset_validator_reports_qa_incomplete(tmp_path: Path) -> None:
    root = tmp_path / "sample"
    seq_dir = root / "sequences"
    img_dir = root / "imagery" / "seq-1"
    seq_dir.mkdir(parents=True)
    img_dir.mkdir(parents=True)
    for folder in ("reference_dtm", "ground_truth", "outputs", "qa_reports"):
        (root / folder).mkdir()
    (img_dir / "img-1.jpg").write_bytes(b"not checked")
    (img_dir / "img-2.jpg").write_bytes(b"not checked")
    (seq_dir / "filtered_sequences.json").write_text(
        json.dumps(
            {
                "sequence_details": {
                    "seq-1": {
                        "frame_count": 2,
                    }
                }
            }
        ),
        encoding="utf8",
    )
    result = validate_sample_dataset(root, expected_sequences=1, expected_images=2)
    assert result["ok"] is True
    assert result["qa_complete"] is False
    assert result["warnings"]


def test_cli_exposes_run_subcommand() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "dtm_from_mapillary.cli.pipeline", "--help"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    assert "run" in result.stdout


def test_fixture_pipeline_smoke_produces_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = tmp_path / "dataset"
    seq_dir = dataset / "sequences"
    seq_dir.mkdir(parents=True)
    for folder in ("imagery", "reference_dtm", "ground_truth", "outputs", "qa_reports"):
        (dataset / folder).mkdir()
    bbox = "-48.0001,-27.0001,-47.9998,-26.9998"
    (dataset / "config.json").write_text(
        json.dumps(
            {
                "bbox_string": bbox,
                "statistics": {"car_sequences": 1, "cached_images": 0},
            }
        ),
        encoding="utf8",
    )
    (seq_dir / "metadata.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    _geojson_image("img-1", "seq-1", -48.0, -27.0, 10.0, 0),
                    _geojson_image("img-2", "seq-1", -47.99995, -26.99995, 10.1, 1000),
                    _geojson_image("img-3", "seq-1", -47.99990, -26.99990, 10.2, 2000),
                ],
            }
        ),
        encoding="utf8",
    )
    monkeypatch.setattr(
        "dtm_from_mapillary.cli.pipeline.corridor_from_osm_bbox",
        lambda bbox: {
            "geometry": {"bbox": bbox},
            "crs": "EPSG:4326",
            "source": "test-rectangle",
            "buffer_m": 25.0,
            "geometry_type": "rectangle",
        },
    )
    monkeypatch.setenv("OPEN_SFM_FORCE_SYNTHETIC", "1")
    monkeypatch.setenv("COLMAP_FORCE_SYNTHETIC", "1")
    monkeypatch.setattr(
        "dtm_from_mapillary.cli.pipeline.consensus_agree",
        lambda pts_a, pts_b, pts_c: [
            {
                "x": 0.0,
                "y": 0.0,
                "z": 10.0,
                "sources": ["A", "B"],
                "support": 2,
                "sem_prob": 0.9,
                "uncertainty": 0.2,
            },
            {
                "x": 0.5,
                "y": 0.0,
                "z": 10.1,
                "sources": ["A", "B"],
                "support": 2,
                "sem_prob": 0.9,
                "uncertainty": 0.2,
            },
        ],
    )

    manifest = run_pipeline(
        dataset_dir=str(dataset),
        out_dir=str(dataset / "outputs"),
        allow_synthetic=True,
        strict_production=False,
    )

    assert manifest["ingestion"]["sequence_count"] == 1
    assert manifest["ground_points"]["exported"] > 0
    assert Path(manifest["outputs"]["geotiffs"]["dtm_0p5m_ellipsoid.tif"]).exists()
    assert Path(manifest["outputs"]["agreement_maps"]).exists()
    assert Path(dataset / "outputs" / "report.html").exists()


def _geojson_image(
    image_id: str,
    seq_id: str,
    lon: float,
    lat: float,
    alt: float,
    captured_at: int,
) -> dict:
    return {
        "type": "Feature",
        "properties": {
            "id": image_id,
            "sequence": seq_id,
            "captured_at": captured_at,
            "altitude": alt,
            "camera_type": "perspective",
            "width": 1920,
            "height": 1080,
        },
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
    }
