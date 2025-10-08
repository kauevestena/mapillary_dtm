"""Tests for breakline integration and 3D projection."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ground.breakline_integration import (
    Breakline3D,
    densify_breaklines,
    merge_breakline_segments,
    project_curbs_to_3d,
    simplify_breaklines,
    _douglas_peucker_3d,
    _point_line_distance_3d,
    _polyline_length,
    _resample_polyline_3d,
    _segments_overlap,
)


class MockCurbLine:
    """Mock curb line for testing."""

    def __init__(self, image_id: str, xy_norm: list, confidence: float = 0.8):
        self.image_id = image_id
        self.xy_norm = xy_norm
        self.confidence = confidence


def test_polyline_length():
    """Test 3D polyline length calculation."""
    # Simple horizontal line
    points = [(0.0, 0.0, 0.0), (3.0, 4.0, 0.0)]
    length = _polyline_length(points)
    assert abs(length - 5.0) < 1e-6

    # 3D line
    points = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 1.0, 1.0)]
    length = _polyline_length(points)
    assert abs(length - 3.0) < 1e-6

    # Empty polyline
    assert _polyline_length([]) == 0.0
    assert _polyline_length([(0.0, 0.0, 0.0)]) == 0.0


def test_segments_overlap():
    """Test segment overlap detection."""
    pts1 = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
    pts2 = [(10.2, 0.0, 0.0), (20.0, 0.0, 0.0)]

    # Should overlap with threshold 0.3
    assert _segments_overlap(pts1, pts2, 0.3)

    # Should not overlap with threshold 0.1
    assert not _segments_overlap(pts1, pts2, 0.1)

    # Parallel segments far apart
    pts3 = [(0.0, 5.0, 0.0), (10.0, 5.0, 0.0)]
    assert not _segments_overlap(pts1, pts3, 0.3)


def test_point_line_distance_3d():
    """Test perpendicular distance calculation."""
    # Point above midpoint of horizontal line
    point = np.array([5.0, 5.0, 0.0])
    line_start = np.array([0.0, 0.0, 0.0])
    line_end = np.array([10.0, 0.0, 0.0])

    dist = _point_line_distance_3d(point, line_start, line_end)
    assert abs(dist - 5.0) < 1e-6

    # Point at line start
    point = np.array([0.0, 0.0, 0.0])
    dist = _point_line_distance_3d(point, line_start, line_end)
    assert abs(dist) < 1e-6

    # Point beyond line end (should project to endpoint)
    point = np.array([15.0, 5.0, 0.0])
    dist = _point_line_distance_3d(point, line_start, line_end)
    expected = math.sqrt(5**2 + 5**2)
    assert abs(dist - expected) < 1e-6


def test_douglas_peucker_3d():
    """Test Douglas-Peucker simplification."""
    # Straight line - should keep only endpoints
    points = [(float(i), 0.0, 0.0) for i in range(10)]
    simplified = _douglas_peucker_3d(points, tolerance=0.1)
    assert len(simplified) == 2
    assert simplified[0] == points[0]
    assert simplified[-1] == points[-1]

    # Line with spike - should keep the spike
    points = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (5.0, 3.0, 0.0), (10.0, 0.0, 0.0)]
    simplified = _douglas_peucker_3d(points, tolerance=0.1)
    assert len(simplified) > 2  # Should keep the spike point

    # Already simple polyline
    points = [(0.0, 0.0, 0.0), (10.0, 10.0, 10.0)]
    simplified = _douglas_peucker_3d(points, tolerance=1.0)
    assert simplified == points


def test_resample_polyline_3d():
    """Test uniform polyline resampling."""
    # Line from 0 to 10 with 0.5m spacing
    points = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
    resampled = _resample_polyline_3d(points, max_spacing=0.5)

    # Should have ~20 points
    assert len(resampled) >= 20

    # Check spacing
    for i in range(len(resampled) - 1):
        p0 = np.array(resampled[i])
        p1 = np.array(resampled[i + 1])
        dist = np.linalg.norm(p1 - p0)
        assert dist <= 0.5 + 1e-6

    # First and last should match original
    assert resampled[0] == points[0]
    assert resampled[-1] == points[-1]


def test_project_curbs_to_3d_basic():
    """Test basic curb projection to 3D."""
    # Create synthetic curb detection
    curbs = {
        "seq1": [
            MockCurbLine(
                image_id="img1",
                xy_norm=[(0.3, 0.5), (0.4, 0.5), (0.5, 0.5)],  # Horizontal line
                confidence=0.85,
            )
        ]
    }

    # Create synthetic camera pose (identity rotation, camera at origin)
    camera_poses = {
        "img1": {
            "rotation": np.eye(3).tolist(),
            "translation": [0.0, 0.0, 1.5],  # Camera 1.5m above ground
        }
    }

    camera_models = {"img1": {"width": 1920, "height": 1080, "focal": 0.8}}

    # Ground plane at z=0
    consensus_points = [
        {"x": 0.0, "y": 0.0, "z": 0.0},
        {"x": 10.0, "y": 0.0, "z": 0.0},
        {"x": 0.0, "y": 10.0, "z": 0.0},
    ]

    breaklines = project_curbs_to_3d(
        curbs=curbs,
        camera_poses=camera_poses,
        camera_models=camera_models,
        consensus_points=consensus_points,
        max_height_dev_m=1.0,
    )

    assert len(breaklines) == 1
    bl = breaklines[0]
    assert bl.seq_id == "seq1"
    assert bl.type == "curb"
    assert len(bl.points) == 3

    # All points should be near ground (z ≈ 0)
    for x, y, z in bl.points:
        assert abs(z) < 0.5


def test_project_curbs_missing_camera():
    """Test curb projection with missing camera data."""
    curbs = {
        "seq1": [
            MockCurbLine(image_id="img_missing", xy_norm=[(0.5, 0.5)], confidence=0.8)
        ]
    }

    # Empty camera data
    breaklines = project_curbs_to_3d(
        curbs=curbs, camera_poses={}, camera_models={}, consensus_points=[]
    )

    # Should return empty list (no valid projections)
    assert len(breaklines) == 0


def test_merge_breakline_segments():
    """Test breakline segment merging."""
    # Two overlapping segments
    bl1 = Breakline3D(
        seq_id="seq1",
        type="curb",
        points=[(0.0, 0.0, 0.0), (5.0, 0.0, 0.0)],
        confidence=0.8,
        image_ids=["img1"],
    )

    bl2 = Breakline3D(
        seq_id="seq1",
        type="curb",
        points=[(5.1, 0.0, 0.0), (10.0, 0.0, 0.0)],
        confidence=0.9,
        image_ids=["img2"],
    )

    # Should merge with default threshold
    merged = merge_breakline_segments([bl1, bl2], merge_dist_m=0.5, min_length_m=1.0)

    assert len(merged) == 1
    assert merged[0].confidence > 0.8  # Averaged confidence


def test_merge_short_segments_filtered():
    """Test that short segments are filtered out."""
    short_bl = Breakline3D(
        seq_id="seq1",
        type="curb",
        points=[(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)],  # 0.5m long
        confidence=0.8,
        image_ids=["img1"],
    )

    merged = merge_breakline_segments([short_bl], min_length_m=2.0)

    # Should be filtered out
    assert len(merged) == 0


def test_simplify_breaklines():
    """Test breakline simplification."""
    # Create zigzag polyline
    points = [(0.0, 0.0, 0.0)]
    for i in range(1, 20):
        y = 0.01 if i % 2 == 0 else -0.01  # Small zigzag
        points.append((float(i), y, 0.0))

    bl = Breakline3D(
        seq_id="seq1", type="curb", points=points, confidence=0.8, image_ids=["img1"]
    )

    simplified = simplify_breaklines([bl], tolerance_m=0.05)

    assert len(simplified) == 1
    # Should have fewer points after simplification
    assert len(simplified[0].points) < len(points)


def test_densify_breaklines():
    """Test breakline densification."""
    # Long segment (10m) with only 2 points
    bl = Breakline3D(
        seq_id="seq1",
        type="curb",
        points=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
        confidence=0.8,
        image_ids=["img1"],
    )

    vertices, edges = densify_breaklines([bl], max_spacing_m=0.5)

    # Should have ~20 vertices
    assert vertices.shape[0] >= 20
    assert vertices.shape[1] == 3  # XYZ

    # Should have 19 edges (connecting consecutive points)
    assert len(edges) == vertices.shape[0] - 1

    # Verify edge connectivity
    for i, (start, end) in enumerate(edges):
        assert end == start + 1  # Consecutive vertices


def test_densify_multiple_breaklines():
    """Test densification with multiple breaklines."""
    bl1 = Breakline3D(
        seq_id="seq1",
        type="curb",
        points=[(0.0, 0.0, 0.0), (5.0, 0.0, 0.0)],
        confidence=0.8,
        image_ids=["img1"],
    )

    bl2 = Breakline3D(
        seq_id="seq2",
        type="lane_edge",
        points=[(10.0, 0.0, 0.0), (15.0, 0.0, 0.0)],
        confidence=0.9,
        image_ids=["img2"],
    )

    vertices, edges = densify_breaklines([bl1, bl2], max_spacing_m=1.0)

    # Should have vertices from both breaklines
    assert vertices.shape[0] > 10

    # Edges should not connect between different breaklines
    for i, (start, end) in enumerate(edges):
        # Check if edge crosses breakline boundary
        # (simple check: consecutive edges should increment by 1)
        if i > 0:
            prev_end = edges[i - 1][1]
            if start != prev_end:
                # This is the start of a new breakline
                assert start == prev_end + 1


def test_empty_breaklines():
    """Test handling of empty breakline lists."""
    # Empty list
    merged = merge_breakline_segments([])
    assert merged == []

    simplified = simplify_breaklines([])
    assert simplified == []

    vertices, edges = densify_breaklines([])
    assert vertices.shape[0] == 0
    assert edges == []


def test_breakline3d_dataclass():
    """Test Breakline3D dataclass creation."""
    bl = Breakline3D(
        seq_id="seq1",
        type="median",
        points=[(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
        confidence=0.95,
        image_ids=["img1", "img2", "img3"],
    )

    assert bl.seq_id == "seq1"
    assert bl.type == "median"
    assert len(bl.points) == 2
    assert bl.confidence == 0.95
    assert len(bl.image_ids) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
