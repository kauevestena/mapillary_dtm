"""Tests for OpenSfM self-calibration integration."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if "dtm_from_mapillary" not in sys.modules:
    pkg = types.ModuleType("dtm_from_mapillary")
    pkg.__path__ = [str(ROOT)]
    sys.modules["dtm_from_mapillary"] = pkg
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dtm_from_mapillary.common_core import FrameMeta, Pose
from tests.sample_loader import get_sample_frames
from dtm_from_mapillary.geom.sfm_opensfm import (
    run,
    _extract_correspondences_for_frame,
    _refine_sequence_cameras,
    _camera_from_frame,
)


def _build_frames(seq_id: str, n_frames: int = 4) -> list[FrameMeta]:
    """Build test frames with camera parameters."""
    return [
        FrameMeta(
            image_id=f"{seq_id}-frame-{i}",
            seq_id=seq_id,
            captured_at_ms=1_700_000_000_000 + i * 100,
            lon=-48.596644 + 0.0001 * i,
            lat=-27.591363 + 0.0001 * i,
            alt_ellip=10.0 + 0.1 * i,
            camera_type="perspective",
            cam_params={
                "width": 4000,
                "height": 3000,
                "focal": 1.0,  # Default focal length
                "principal_point": [0.5, 0.5],  # Exact center
                "k1": 0.0,
                "k2": 0.0,
            },
            quality_score=0.9,
        )
        for i in range(n_frames)
    ]


def test_opensfm_run_without_refinement():
    seqs, imagery_root = get_sample_frames()
    results = run(seqs, imagery_root=imagery_root)
    assert list(seqs.keys())[0] in results

def test_opensfm_run_with_full_refinement():
    seqs, imagery_root = get_sample_frames()
    results = run(seqs, imagery_root=imagery_root, refine_cameras=True, refinement_method="full")
    assert list(seqs.keys())[0] in results

def test_opensfm_run_with_quick_refinement():
    seqs, imagery_root = get_sample_frames()
    results = run(seqs, imagery_root=imagery_root, refine_cameras=True, refinement_method="quick")
    assert list(seqs.keys())[0] in results

def test_opensfm_refinement_insufficient_points():
    seqs, imagery_root = get_sample_frames()
    from dtm_from_mapillary.geom.opensfm_adapter import OpenSfMUnavailable
    try:
        results = run(seqs, imagery_root=imagery_root, refine_cameras=True)
        assert list(seqs.keys())[0] in results
    except OpenSfMUnavailable:
        pass # Fine if it raises
        
def test_extract_correspondences_for_frame():
    """Test correspondence extraction for a single frame."""
    frame = _build_frames("seq1", n_frames=1)[0]

    # Create synthetic pose and points
    pose = Pose(
        R=np.eye(3),
        t=np.array([0.0, 0.0, 0.0]),
    )

    # Points in front of camera (positive Z in camera frame = world frame here)
    points_xyz = np.array(
        [
            [1.0, 0.0, 5.0],
            [0.0, 1.0, 5.0],
            [-1.0, 0.0, 5.0],
            [0.0, -1.0, 5.0],
            [0.5, 0.5, 3.0],
        ]
    )

    rng = np.random.default_rng(42)
    points_3d, points_2d = _extract_correspondences_for_frame(
        frame, pose, points_xyz, rng
    )

    # Should get all 5 points (all in front)
    assert points_3d.shape == (5, 3)
    assert points_2d.shape == (5, 2)

    # Check normalized coordinates are reasonable (< 1.0 typically for perspective)
    assert np.all(np.abs(points_2d) < 2.0)


def test_extract_correspondences_filters_behind_camera():
    """Test that points behind camera are filtered out."""
    frame = _build_frames("seq1", n_frames=1)[0]

    pose = Pose(R=np.eye(3), t=np.array([0.0, 0.0, 0.0]))

    # Mix of points in front and behind camera
    points_xyz = np.array(
        [
            [0.0, 0.0, 5.0],  # Front (Z > 0)
            [0.0, 0.0, -5.0],  # Behind (Z < 0)
            [1.0, 0.0, 3.0],  # Front
            [0.0, 0.0, -2.0],  # Behind
        ]
    )

    rng = np.random.default_rng(42)
    points_3d, points_2d = _extract_correspondences_for_frame(
        frame, pose, points_xyz, rng
    )

    # Should get only 2 points (those in front)
    assert points_3d.shape == (2, 3)
    assert points_2d.shape == (2, 2)

    # Verify Z coordinates are positive for returned points
    points_cam = (points_3d - pose.t) @ pose.R
    assert np.all(points_cam[:, 2] > 0)


def test_extract_correspondences_limits_max_points():
    """Test that max_points limit is respected."""
    frame = _build_frames("seq1", n_frames=1)[0]
    pose = Pose(R=np.eye(3), t=np.array([0.0, 0.0, 0.0]))

    # Create many points (200 in front of camera)
    points_xyz = np.random.randn(200, 3)
    points_xyz[:, 2] = np.abs(points_xyz[:, 2]) + 1.0  # Ensure positive Z

    rng = np.random.default_rng(42)
    points_3d, points_2d = _extract_correspondences_for_frame(
        frame, pose, points_xyz, rng, max_points=50
    )

    # Should be limited to 50
    assert points_3d.shape == (50, 3)
    assert points_2d.shape == (50, 2)


def test_extract_correspondences_empty_points():
    """Test handling of empty point cloud."""
    frame = _build_frames("seq1", n_frames=1)[0]
    pose = Pose(R=np.eye(3), t=np.array([0.0, 0.0, 0.0]))

    points_xyz = np.zeros((0, 3))

    rng = np.random.default_rng(42)
    points_3d, points_2d = _extract_correspondences_for_frame(
        frame, pose, points_xyz, rng
    )

    # Should return empty arrays
    assert points_3d.shape == (0, 3)
    assert points_2d.shape == (0, 2)


def test_camera_from_frame_basic():
    """Test camera parameter extraction from FrameMeta."""
    frame = FrameMeta(
        image_id="img1",
        seq_id="seq1",
        captured_at_ms=1700000000000,
        lon=-48.5,
        lat=-27.5,
        alt_ellip=10.0,
        camera_type="perspective",
        cam_params={
            "focal": 0.85,
            "principal_point": [0.48, 0.52],
            "k1": -0.05,
            "k2": 0.01,
        },
        quality_score=0.9,
    )

    camera = _camera_from_frame(frame)

    assert camera["focal"] == 0.85
    assert camera["principal_point"] == [0.48, 0.52]
    assert camera["projection_type"] == "perspective"
    assert camera["k1"] == -0.05
    assert camera["k2"] == 0.01


def test_camera_from_frame_defaults():
    """Test camera extraction with missing parameters (defaults)."""
    frame = FrameMeta(
        image_id="img1",
        seq_id="seq1",
        captured_at_ms=1700000000000,
        lon=-48.5,
        lat=-27.5,
        alt_ellip=10.0,
        camera_type="fisheye",
        cam_params={
            "width": 4000,
            "height": 3000,
        },
        quality_score=0.9,
    )

    camera = _camera_from_frame(frame)

    # Should get defaults
    assert camera["focal"] == 1.0
    assert camera["principal_point"] == [0.5, 0.5]
    assert camera["projection_type"] == "fisheye"


def test_camera_from_frame_cx_cy_format():
    """Test camera extraction when using cx/cy instead of principal_point."""
    frame = FrameMeta(
        image_id="img1",
        seq_id="seq1",
        captured_at_ms=1700000000000,
        lon=-48.5,
        lat=-27.5,
        alt_ellip=10.0,
        camera_type="perspective",
        cam_params={
            "focal": 0.9,
            "cx": 0.49,
            "cy": 0.51,
        },
        quality_score=0.9,
    )

    camera = _camera_from_frame(frame)

    assert camera["focal"] == 0.9
    assert camera["principal_point"] == [0.49, 0.51]


def test_refine_sequence_cameras_integration():
    """Test full sequence refinement workflow."""
    seqs, _ = get_sample_frames()
    frames = list(seqs.values())[0][:3]

    # Create synthetic poses and points
    poses = {}
    points_list = []

    for i, frame in enumerate(frames):
        # Simple identity rotation, translated camera positions
        poses[frame.image_id] = Pose(
            R=np.eye(3),
            t=np.array([i * 2.0, 0.0, 0.0]),
        )

        # Add points near each camera
        for j in range(10):
            points_list.append(
                [
                    i * 2.0 + np.random.randn() * 0.5,
                    np.random.randn() * 0.5,
                    5.0 + np.random.randn() * 0.5,
                ]
            )

    points_xyz = np.array(points_list)
    rng = np.random.default_rng(42)

    # Call refinement
    refined_frames, metadata = _refine_sequence_cameras(
        frames, poses, points_xyz, method="quick", rng=rng
    )

    # Should succeed
    assert len(refined_frames) == 3
    assert metadata.get("refined_count", 0) > 0

    # Check that camera parameters were updated
    for frame in refined_frames:
        assert "focal" in frame.cam_params
        assert "principal_point" in frame.cam_params


def test_refine_sequence_cameras_insufficient_correspondences():
    """Test graceful handling when frames have insufficient correspondences."""
    seqs, _ = get_sample_frames()
    frames = list(seqs.values())[0][:2]

    poses = {
        frame.image_id: Pose(R=np.eye(3), t=np.array([0.0, 0.0, 0.0]))
        for frame in frames
    }

    # Very few points
    points_xyz = np.array([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]])

    rng = np.random.default_rng(42)
    refined_frames, metadata = _refine_sequence_cameras(
        frames, poses, points_xyz, method="quick", rng=rng
    )

    # Should handle gracefully (might have error or 0 refined)
    assert len(refined_frames) == 2
    # Refined count might be 0 due to insufficient points
    assert "refined_count" in metadata or "error" in metadata


def test_opensfm_backward_compatibility():
    seqs, imagery_root = get_sample_frames()
    results = run(seqs, imagery_root=imagery_root, rng_seed=42)

    # Should work with default refine_cameras=False
    seq_id = list(seqs.keys())[0]
    assert seq_id in results
    assert not results[seq_id].metadata.get("cameras_refined", False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
