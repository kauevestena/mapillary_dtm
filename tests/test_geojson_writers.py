"""
Tests for io.geojson_writers.

Covers:
- write_frames_geojson: valid GeoJSON output, all frames present, required props
- write_camera_positions_geojson: ENU -> WGS84 conversion, source label
- write_all_camera_positions_geojson: multi-source merge
- FrameMeta.to_geojson_feature: geometry type, coordinate count with/without alt
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dtm_from_mapillary.common_core import FrameMeta, Pose, ReconstructionResult
from dtm_from_mapillary.io.geojson_writers import (
    write_all_camera_positions_geojson,
    write_camera_positions_geojson,
    write_frames_geojson,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_frame(
    image_id: str = "img001",
    seq_id: str = "seq001",
    lon: float = -49.276,
    lat: float = -25.443,
    alt: float | None = 925.0,
    camera_type: str = "perspective",
) -> FrameMeta:
    return FrameMeta(
        image_id=image_id,
        seq_id=seq_id,
        captured_at_ms=1_600_000_000_000,
        lon=lon,
        lat=lat,
        alt_ellip=alt,
        camera_type=camera_type,
        cam_params={"focal": 0.8, "width": 1600, "height": 1200},
        quality_score=0.85,
        thumbnail_url="https://example.com/img.jpg",
    )


def _make_recon(seq_id: str = "seq001", n_frames: int = 3) -> ReconstructionResult:
    frames = [_make_frame(image_id=f"img{i:03d}", seq_id=seq_id) for i in range(n_frames)]
    poses = {}
    rng = np.random.default_rng(42)
    for frame in frames:
        R = np.eye(3)
        t = rng.uniform(-5.0, 5.0, size=(3,))
        poses[frame.image_id] = Pose(R=R, t=t)
    points_xyz = rng.uniform(0.0, 50.0, size=(10, 3))
    return ReconstructionResult(
        seq_id=seq_id,
        frames=frames,
        poses=poses,
        points_xyz=points_xyz,
        source="test",
        metadata={"coordinate_frame": "enu"},
    )


# ---------------------------------------------------------------------------
# FrameMeta.to_geojson_feature
# ---------------------------------------------------------------------------


class TestToGeojsonFeature:
    def test_returns_feature_dict(self):
        f = _make_frame()
        feat = f.to_geojson_feature()
        assert feat["type"] == "Feature"

    def test_geometry_is_point(self):
        f = _make_frame()
        geom = f.to_geojson_feature()["geometry"]
        assert geom["type"] == "Point"

    def test_coordinates_with_altitude(self):
        f = _make_frame(lon=-49.276, lat=-25.443, alt=925.0)
        coords = f.to_geojson_feature()["geometry"]["coordinates"]
        assert len(coords) == 3
        assert coords[0] == pytest.approx(-49.276)
        assert coords[1] == pytest.approx(-25.443)
        assert coords[2] == pytest.approx(925.0)

    def test_coordinates_without_altitude(self):
        f = _make_frame(alt=None)
        coords = f.to_geojson_feature()["geometry"]["coordinates"]
        assert len(coords) == 2

    def test_required_properties(self):
        f = _make_frame(image_id="abc", seq_id="s1")
        props = f.to_geojson_feature()["properties"]
        assert props["image_id"] == "abc"
        assert props["seq_id"] == "s1"
        assert "captured_at_ms" in props
        assert "camera_type" in props
        assert "cam_params" in props
        assert "quality_score" in props


# ---------------------------------------------------------------------------
# write_frames_geojson
# ---------------------------------------------------------------------------


class TestWriteFramesGeojson:
    def test_creates_file(self, tmp_path):
        seqs = {"s1": [_make_frame()]}
        out = tmp_path / "out.geojson"
        result = write_frames_geojson(seqs, out)
        assert Path(result).exists()

    def test_valid_geojson(self, tmp_path):
        seqs = {"s1": [_make_frame("i1"), _make_frame("i2")]}
        out = tmp_path / "frames.geojson"
        write_frames_geojson(seqs, out)
        data = json.loads(out.read_text())
        assert data["type"] == "FeatureCollection"

    def test_feature_count(self, tmp_path):
        seqs = {
            "s1": [_make_frame("i1"), _make_frame("i2")],
            "s2": [_make_frame("i3")],
        }
        out = tmp_path / "frames.geojson"
        write_frames_geojson(seqs, out)
        data = json.loads(out.read_text())
        assert len(data["features"]) == 3

    def test_all_features_are_points(self, tmp_path):
        seqs = {"s1": [_make_frame("i1"), _make_frame("i2")]}
        out = tmp_path / "frames.geojson"
        write_frames_geojson(seqs, out)
        data = json.loads(out.read_text())
        for feat in data["features"]:
            assert feat["geometry"]["type"] == "Point"

    def test_properties_contain_image_id(self, tmp_path):
        seqs = {"s1": [_make_frame("unique_img_id")]}
        out = tmp_path / "frames.geojson"
        write_frames_geojson(seqs, out)
        data = json.loads(out.read_text())
        ids = [f["properties"]["image_id"] for f in data["features"]]
        assert "unique_img_id" in ids

    def test_crs_field_present(self, tmp_path):
        seqs = {"s1": [_make_frame()]}
        out = tmp_path / "frames.geojson"
        write_frames_geojson(seqs, out)
        data = json.loads(out.read_text())
        assert "crs" in data

    def test_empty_sequences(self, tmp_path):
        out = tmp_path / "frames.geojson"
        write_frames_geojson({}, out)
        data = json.loads(out.read_text())
        assert data["features"] == []

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "nested" / "deep" / "frames.geojson"
        write_frames_geojson({"s1": [_make_frame()]}, out)
        assert out.exists()

    def test_returns_str_path(self, tmp_path):
        out = tmp_path / "frames.geojson"
        result = write_frames_geojson({"s1": [_make_frame()]}, out)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# write_camera_positions_geojson
# ---------------------------------------------------------------------------


class TestWriteCameraPositionsGeojson:
    # Use a well-known ENU origin in Brazil (same region as the sample data)
    LON0, LAT0, H0 = -49.276, -25.443, 925.0

    def test_creates_file(self, tmp_path):
        recon = {"seq001": _make_recon()}
        out = tmp_path / "cameras.geojson"
        result = write_camera_positions_geojson(recon, self.LON0, self.LAT0, self.H0, out)
        assert Path(result).exists()

    def test_valid_geojson(self, tmp_path):
        recon = {"seq001": _make_recon()}
        out = tmp_path / "cameras.geojson"
        write_camera_positions_geojson(recon, self.LON0, self.LAT0, self.H0, out)
        data = json.loads(out.read_text())
        assert data["type"] == "FeatureCollection"

    def test_feature_count_matches_poses(self, tmp_path):
        recon = {"seq001": _make_recon(n_frames=5)}
        out = tmp_path / "cameras.geojson"
        write_camera_positions_geojson(recon, self.LON0, self.LAT0, self.H0, out)
        data = json.loads(out.read_text())
        assert len(data["features"]) == 5

    def test_source_label_in_properties(self, tmp_path):
        recon = {"seq001": _make_recon()}
        out = tmp_path / "cameras.geojson"
        write_camera_positions_geojson(
            recon, self.LON0, self.LAT0, self.H0, out, source_label="opensfm"
        )
        data = json.loads(out.read_text())
        for feat in data["features"]:
            assert feat["properties"]["source"] == "opensfm"

    def test_gnss_fields_present(self, tmp_path):
        recon = {"seq001": _make_recon()}
        out = tmp_path / "cameras.geojson"
        write_camera_positions_geojson(recon, self.LON0, self.LAT0, self.H0, out)
        data = json.loads(out.read_text())
        for feat in data["features"]:
            assert "gnss_lon" in feat["properties"]
            assert "gnss_lat" in feat["properties"]

    def test_coordinates_are_three_dimensional(self, tmp_path):
        recon = {"seq001": _make_recon()}
        out = tmp_path / "cameras.geojson"
        write_camera_positions_geojson(recon, self.LON0, self.LAT0, self.H0, out)
        data = json.loads(out.read_text())
        for feat in data["features"]:
            assert len(feat["geometry"]["coordinates"]) == 3

    def test_positions_near_origin(self, tmp_path):
        """Cameras within ±50 m of origin should end up within ~0.001 deg."""
        recon = {"seq001": _make_recon()}
        out = tmp_path / "cameras.geojson"
        write_camera_positions_geojson(recon, self.LON0, self.LAT0, self.H0, out)
        data = json.loads(out.read_text())
        for feat in data["features"]:
            lon, lat, _ = feat["geometry"]["coordinates"]
            assert abs(lon - self.LON0) < 0.01
            assert abs(lat - self.LAT0) < 0.01


# ---------------------------------------------------------------------------
# write_all_camera_positions_geojson
# ---------------------------------------------------------------------------


class TestWriteAllCameraPositionsGeojson:
    LON0, LAT0, H0 = -49.276, -25.443, 925.0

    def test_combines_sources(self, tmp_path):
        reconA = {"seq001": _make_recon("seq001", n_frames=3)}
        reconB = {"seq002": _make_recon("seq002", n_frames=2)}
        out = tmp_path / "all_cameras.geojson"
        write_all_camera_positions_geojson(
            {"opensfm": reconA, "colmap": reconB},
            self.LON0, self.LAT0, self.H0,
            out,
        )
        data = json.loads(out.read_text())
        assert len(data["features"]) == 5

    def test_source_labels_preserved(self, tmp_path):
        reconA = {"seq001": _make_recon("seq001", n_frames=2)}
        reconB = {"seq002": _make_recon("seq002", n_frames=1)}
        out = tmp_path / "all_cameras.geojson"
        write_all_camera_positions_geojson(
            {"opensfm": reconA, "colmap": reconB},
            self.LON0, self.LAT0, self.H0,
            out,
        )
        data = json.loads(out.read_text())
        sources = {f["properties"]["source"] for f in data["features"]}
        assert "opensfm" in sources
        assert "colmap" in sources

    def test_empty_reconstruction_skipped(self, tmp_path):
        reconA = {"seq001": _make_recon("seq001", n_frames=2)}
        out = tmp_path / "all_cameras.geojson"
        write_all_camera_positions_geojson(
            {"opensfm": reconA, "colmap": {}},
            self.LON0, self.LAT0, self.H0,
            out,
        )
        data = json.loads(out.read_text())
        assert len(data["features"]) == 2

    def test_valid_geojson_output(self, tmp_path):
        recon = {"seq001": _make_recon()}
        out = tmp_path / "all_cameras.geojson"
        write_all_camera_positions_geojson(
            {"vo": recon}, self.LON0, self.LAT0, self.H0, out
        )
        data = json.loads(out.read_text())
        assert data["type"] == "FeatureCollection"
        assert "crs" in data
