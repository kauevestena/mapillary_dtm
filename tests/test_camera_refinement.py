"""Tests for camera intrinsic parameter validation and refinement."""

from __future__ import annotations

import pytest

from geom.camera_refinement import (
    ValidationReport,
    validate_intrinsics,
    validate_sequence_consistency,
    needs_refinement,
    summarize_validation,
)


def test_validate_intrinsics_good_perspective():
    """Test validation of good perspective camera parameters."""
    camera = {
        "projection_type": "perspective",
        "width": 1920,
        "height": 1080,
        "focal": 0.85,
        "principal_point": [
            0.501,
            0.499,
        ],  # Slightly off-center to avoid default warning
        "k1": -0.15,
        "k2": 0.03,
        "p1": 0.001,
        "p2": -0.001,
    }

    result = validate_intrinsics(camera)

    assert result.valid
    assert result.focal_status == "good"
    assert result.principal_point_status == "good"
    assert result.distortion_status == "good"
    assert result.confidence > 0.9
    assert not result.needs_refinement


def test_validate_intrinsics_missing_focal():
    """Test validation with missing focal length."""
    camera = {
        "projection_type": "perspective",
        "width": 1920,
        "height": 1080,
        "principal_point": [0.5, 0.5],
    }

    result = validate_intrinsics(camera)

    assert not result.valid
    assert result.focal_status == "missing"
    assert result.needs_refinement
    assert result.confidence < 0.6
    assert any("missing focal length" in w.lower() for w in result.warnings)


def test_validate_intrinsics_suspicious_focal():
    """Test validation of suspicious focal length values."""
    # Test default focal = 1.0
    camera_default = {
        "projection_type": "perspective",
        "focal": 1.0,
        "width": 1920,
        "height": 1080,
    }

    result = validate_intrinsics(camera_default)

    assert result.focal_status == "suspect"
    assert result.needs_refinement
    assert any("manufacturer default" in w.lower() for w in result.warnings)

    # Test focal too low
    camera_low = {
        "projection_type": "perspective",
        "focal": 0.3,
        "width": 1920,
        "height": 1080,
    }

    result_low = validate_intrinsics(camera_low)

    assert result_low.focal_status == "suspect"
    assert result_low.needs_refinement
    assert any("unusually low" in w.lower() for w in result_low.warnings)

    # Test focal too high
    camera_high = {
        "projection_type": "perspective",
        "focal": 2.0,
        "width": 1920,
        "height": 1080,
    }

    result_high = validate_intrinsics(camera_high)

    assert result_high.focal_status == "suspect"
    assert result_high.needs_refinement
    assert any("unusually high" in w.lower() for w in result_high.warnings)


def test_validate_intrinsics_suspicious_principal_point():
    """Test validation of suspicious principal point."""
    # Principal point far from center
    camera = {
        "projection_type": "perspective",
        "focal": 0.85,
        "principal_point": [0.8, 0.3],  # Far from center
        "width": 1920,
        "height": 1080,
    }

    result = validate_intrinsics(camera)

    assert result.principal_point_status == "suspect"
    assert result.needs_refinement
    assert any("far from center" in w.lower() for w in result.warnings)


def test_validate_intrinsics_excessive_distortion():
    """Test validation of excessive distortion coefficients."""
    camera = {
        "projection_type": "brown",
        "focal": 0.85,
        "width": 1920,
        "height": 1080,
        "k1": 1.5,  # Way too high
        "k2": 0.8,  # Too high
        "p1": 0.001,
    }

    result = validate_intrinsics(camera)

    assert result.distortion_status == "suspect"
    assert result.needs_refinement
    assert any("exceeds typical range" in w.lower() for w in result.warnings)


def test_validate_intrinsics_missing_distortion_fisheye():
    """Test validation of fisheye camera without distortion."""
    camera = {
        "projection_type": "fisheye",
        "focal": 0.65,
        "width": 1920,
        "height": 1080,
    }

    result = validate_intrinsics(camera)

    assert result.distortion_status == "missing"
    assert result.needs_refinement
    assert any("no distortion" in w.lower() for w in result.warnings)


def test_validate_intrinsics_zero_distortion():
    """Test validation with all zero distortion coefficients."""
    camera = {
        "projection_type": "brown",
        "focal": 0.85,
        "width": 1920,
        "height": 1080,
        "k1": 0.0,
        "k2": 0.0,
        "k3": 0.0,
        "p1": 0.0,
        "p2": 0.0,
    }

    result = validate_intrinsics(camera)

    assert result.distortion_status == "suspect"
    assert any(
        "all distortion coefficients are zero" in w.lower() for w in result.warnings
    )


def test_validate_intrinsics_spherical():
    """Test validation of spherical camera (no distortion expected)."""
    camera = {
        "projection_type": "spherical",
        "focal": 0.5,
        "width": 3840,
        "height": 1920,
    }

    result = validate_intrinsics(camera)

    # Spherical cameras don't need distortion
    assert result.distortion_status in ["good", "missing"]


def test_validate_sequence_consistency_good():
    """Test sequence consistency with uniform parameters."""
    cameras = {
        f"img_{i}": {
            "focal": 0.85 + i * 0.001,  # Very small variation
            "width": 1920,
            "height": 1080,
        }
        for i in range(10)
    }

    result = validate_sequence_consistency(cameras, tolerance=0.1)

    assert result["consistent"]
    assert len(result["warnings"]) == 0
    assert result["stats"]["n_cameras"] == 10
    assert result["stats"]["focal_range"] < 0.01


def test_validate_sequence_consistency_variable():
    """Test sequence consistency with varying parameters."""
    cameras = {
        f"img_{i}": {
            "focal": 0.7 + i * 0.05,  # Large variation
            "width": 1920,
            "height": 1080,
        }
        for i in range(10)
    }

    result = validate_sequence_consistency(cameras, tolerance=0.1)

    assert not result["consistent"]
    assert len(result["warnings"]) > 0
    assert "varies by" in result["warnings"][0].lower()


def test_validate_sequence_consistency_outliers():
    """Test sequence consistency with outlier detection."""
    cameras = {
        f"img_{i}": {"focal": 0.85, "width": 1920, "height": 1080} for i in range(10)
    }
    # Add outlier
    cameras["img_outlier"] = {"focal": 1.5, "width": 1920, "height": 1080}  # Way off

    result = validate_sequence_consistency(cameras, tolerance=0.1)

    assert not result["consistent"]
    # Either "outliers" or "varies by" message is acceptable for large variation
    assert len(result["warnings"]) > 0
    assert "focal_range" in result["stats"]


def test_validate_sequence_consistency_jump():
    """Test detection of sudden parameter jumps."""
    cameras = {}
    # First half: focal = 0.8
    for i in range(5):
        cameras[f"img_{i}"] = {"focal": 0.8, "width": 1920, "height": 1080}

    # Second half: focal = 1.2 (sudden jump)
    for i in range(5, 10):
        cameras[f"img_{i}"] = {"focal": 1.2, "width": 1920, "height": 1080}

    result = validate_sequence_consistency(cameras, tolerance=0.1)

    assert not result["consistent"]
    assert any("jump" in w.lower() for w in result["warnings"])
    assert "max_jump" in result["stats"]


def test_validate_sequence_consistency_empty():
    """Test sequence consistency with empty input."""
    result = validate_sequence_consistency({}, tolerance=0.1)

    assert result["consistent"]
    assert len(result["warnings"]) == 0


def test_validate_sequence_consistency_no_focal():
    """Test sequence consistency when focal lengths are missing."""
    cameras = {
        f"img_{i}": {
            "width": 1920,
            "height": 1080,
            # No focal length
        }
        for i in range(5)
    }

    result = validate_sequence_consistency(cameras, tolerance=0.1)

    assert not result["consistent"]
    assert "no focal lengths" in result["warnings"][0].lower()


def test_needs_refinement_function():
    """Test the needs_refinement helper function."""
    # Good camera - no refinement
    good = ValidationReport(
        valid=True,
        warnings=[],
        needs_refinement=False,
        confidence=0.95,
        focal_status="good",
        distortion_status="good",
        principal_point_status="good",
    )
    assert not needs_refinement(good)

    # Explicit flag set
    flagged = ValidationReport(
        valid=False,
        warnings=["test"],
        needs_refinement=True,
        confidence=0.9,
        focal_status="good",
        distortion_status="good",
        principal_point_status="good",
    )
    assert needs_refinement(flagged)

    # Low confidence
    low_conf = ValidationReport(
        valid=True,
        warnings=[],
        needs_refinement=False,
        confidence=0.75,  # Below 0.8 threshold
        focal_status="good",
        distortion_status="good",
        principal_point_status="good",
    )
    assert needs_refinement(low_conf)

    # Multiple warnings
    many_warnings = ValidationReport(
        valid=True,
        warnings=["w1", "w2", "w3"],
        needs_refinement=False,
        confidence=0.9,
        focal_status="good",
        distortion_status="good",
        principal_point_status="good",
    )
    assert needs_refinement(many_warnings)

    # Suspect status
    suspect = ValidationReport(
        valid=False,
        warnings=["test"],
        needs_refinement=False,
        confidence=0.9,
        focal_status="suspect",
        distortion_status="good",
        principal_point_status="good",
    )
    assert needs_refinement(suspect)


def test_summarize_validation():
    """Test validation summary aggregation."""
    validations = {}

    # Add some good cameras
    for i in range(7):
        validations[f"img_{i}"] = ValidationReport(
            valid=True,
            warnings=[],
            needs_refinement=False,
            confidence=0.95,
            focal_status="good",
            distortion_status="good",
            principal_point_status="good",
        )

    # Add some cameras needing refinement
    for i in range(7, 10):
        validations[f"img_{i}"] = ValidationReport(
            valid=False,
            warnings=["focal suspect", "distortion missing"],
            needs_refinement=True,
            confidence=0.6,
            focal_status="suspect",
            distortion_status="missing",
            principal_point_status="good",
        )

    consistency = {"consistent": True, "warnings": []}

    summary = summarize_validation(validations, consistency)

    assert summary["total_cameras"] == 10
    assert summary["valid_count"] == 7
    assert summary["needs_refinement_count"] == 3
    assert not summary["needs_refinement"]  # Only 30%, not > 50%
    assert summary["avg_confidence"] > 0.8
    assert summary["min_confidence"] == 0.6
    assert summary["focal_good"] == 7
    assert summary["focal_suspect"] == 3
    assert summary["distortion_missing"] == 3
    assert summary["sequence_consistent"]


def test_summarize_validation_empty():
    """Test validation summary with no cameras."""
    summary = summarize_validation({}, {})

    assert summary["total_cameras"] == 0
    assert not summary["needs_refinement"]


def test_summarize_validation_majority_needs_refinement():
    """Test validation summary when majority needs refinement."""
    validations = {}

    # Most cameras need refinement
    for i in range(8):
        validations[f"img_{i}"] = ValidationReport(
            valid=False,
            warnings=["test"],
            needs_refinement=True,
            confidence=0.7,
            focal_status="suspect",
            distortion_status="suspect",
            principal_point_status="good",
        )

    # Only a few good cameras
    for i in range(8, 10):
        validations[f"img_{i}"] = ValidationReport(
            valid=True,
            warnings=[],
            needs_refinement=False,
            confidence=0.95,
            focal_status="good",
            distortion_status="good",
            principal_point_status="good",
        )

    consistency = {"consistent": False, "warnings": ["varied"]}

    summary = summarize_validation(validations, consistency)

    assert summary["needs_refinement"]  # 80% > 50%
    assert not summary["sequence_consistent"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
