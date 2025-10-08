"""Tests for full self-calibration workflow."""

from __future__ import annotations

import pytest
import numpy as np

from geom.self_calibration import (
    SelfCalibrationResult,
    IterationResult,
    compute_rmse,
    refine_camera_full,
    refine_camera_quick,
    refine_sequence_cameras,
)


def create_calibration_data(
    n_points: int,
    true_focal: float,
    true_pp: tuple,
    true_distortion: dict,
    camera_pose: np.ndarray,
    image_size: tuple = (1920, 1080),
    noise_std: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic data for calibration testing."""
    rng = np.random.RandomState(42)

    # Generate 3D points in camera frame
    points_3d_cam = np.zeros((n_points, 3))
    points_3d_cam[:, 0] = rng.uniform(-5, 5, n_points)
    points_3d_cam[:, 1] = rng.uniform(-3, 3, n_points)
    points_3d_cam[:, 2] = rng.uniform(5, 20, n_points)

    # Transform to world
    points_3d = (camera_pose[:3, :3] @ points_3d_cam.T).T + camera_pose[:3, 3]

    # Project with distortion
    proj_x = points_3d_cam[:, 0] / points_3d_cam[:, 2]
    proj_y = points_3d_cam[:, 1] / points_3d_cam[:, 2]

    # Apply Brown-Conrady distortion
    k1 = true_distortion.get("k1", 0.0)
    k2 = true_distortion.get("k2", 0.0)
    p1 = true_distortion.get("p1", 0.0)
    p2 = true_distortion.get("p2", 0.0)

    r2 = proj_x**2 + proj_y**2
    radial = 1 + k1 * r2 + k2 * r2**2

    x_dist = proj_x * radial + 2 * p1 * proj_x * proj_y + p2 * (r2 + 2 * proj_x**2)
    y_dist = proj_y * radial + p1 * (r2 + 2 * proj_y**2) + 2 * p2 * proj_x * proj_y

    # Convert to pixels
    width, height = image_size
    px = true_focal * width * x_dist + true_pp[0] * width
    py = true_focal * width * y_dist + true_pp[1] * height

    # Add noise
    if noise_std > 0:
        px += rng.normal(0, noise_std, n_points)
        py += rng.normal(0, noise_std, n_points)

    # Normalize
    points_2d = np.column_stack([px / width, py / height])

    return points_3d, points_2d


def test_compute_rmse():
    """Test RMSE computation."""
    true_focal = 0.85
    true_pp = (0.5, 0.5)

    camera_pose = np.eye(4)
    image_size = (1920, 1080)

    points_3d, points_2d = create_calibration_data(
        50, true_focal, true_pp, {}, camera_pose, image_size, noise_std=0.0
    )

    # RMSE with correct parameters should be ~0
    rmse = compute_rmse(
        points_3d, points_2d, true_focal, true_pp, camera_pose, image_size
    )

    assert rmse < 0.1  # Very low with perfect data


def test_compute_rmse_wrong_params():
    """Test RMSE with wrong parameters."""
    true_focal = 0.85
    wrong_focal = 1.0

    camera_pose = np.eye(4)
    image_size = (1920, 1080)

    points_3d, points_2d = create_calibration_data(
        50, true_focal, (0.5, 0.5), {}, camera_pose, image_size
    )

    rmse = compute_rmse(
        points_3d, points_2d, wrong_focal, (0.5, 0.5), camera_pose, image_size
    )

    # Should have significant error
    assert rmse > 5.0


def test_refine_camera_full_focal_only():
    """Test full refinement improving focal length."""
    true_focal = 0.82
    true_pp = (0.5, 0.5)
    true_distortion = {}

    camera_pose = np.eye(4)
    image_size = (1920, 1080)

    points_3d, points_2d = create_calibration_data(
        80, true_focal, true_pp, true_distortion, camera_pose, image_size, noise_std=0.5
    )

    # Start with wrong focal
    camera = {
        "focal": 1.0,
        "principal_point": [0.5, 0.5],
        "projection_type": "perspective",
    }

    result = refine_camera_full(
        points_3d,
        points_2d,
        camera,
        camera_pose,
        image_size,
        max_iterations=3,
        optimize_order=["focal"],  # Only focal
    )

    assert result.converged or result.total_iterations > 0
    assert result.improvement > 0
    assert result.final_rmse < result.validation_report["initial_rmse"]

    # Focal should improve
    assert abs(result.refined_camera["focal"] - true_focal) < 0.15


def test_refine_camera_full_all_parameters():
    """Test full refinement with focal, distortion, and PP."""
    true_focal = 0.88
    true_pp = (0.52, 0.48)
    true_distortion = {"k1": -0.18, "k2": 0.04}

    camera_pose = np.eye(4)
    image_size = (1920, 1080)

    points_3d, points_2d = create_calibration_data(
        100,
        true_focal,
        true_pp,
        true_distortion,
        camera_pose,
        image_size,
        noise_std=0.8,
    )

    # Start with poor estimates
    camera = {
        "focal": 1.0,
        "principal_point": [0.5, 0.5],
        "projection_type": "perspective",
        "k1": 0.0,
        "k2": 0.0,
    }

    result = refine_camera_full(
        points_3d, points_2d, camera, camera_pose, image_size, max_iterations=5
    )

    assert result.total_iterations > 0
    assert result.improvement > 0
    assert result.confidence > 0

    # All parameters should improve
    assert abs(result.refined_camera["focal"] - true_focal) < 0.2
    assert abs(result.refined_camera["principal_point"][0] - true_pp[0]) < 0.1
    assert abs(result.refined_camera["principal_point"][1] - true_pp[1]) < 0.1


def test_refine_camera_full_convergence():
    """Test that refinement converges when parameters are close."""
    true_focal = 0.85
    true_pp = (0.51, 0.49)
    true_distortion = {"k1": -0.15}

    camera_pose = np.eye(4)
    image_size = (1920, 1080)

    points_3d, points_2d = create_calibration_data(
        60, true_focal, true_pp, true_distortion, camera_pose, image_size, noise_std=0.3
    )

    # Start close to truth
    camera = {
        "focal": 0.87,
        "principal_point": [0.5, 0.5],
        "projection_type": "perspective",
        "k1": -0.12,
    }

    result = refine_camera_full(
        points_3d,
        points_2d,
        camera,
        camera_pose,
        image_size,
        convergence_threshold=0.1,
    )

    # Should converge quickly
    assert result.converged or result.total_iterations <= 3


def test_refine_camera_full_insufficient_points():
    """Test that full refinement requires sufficient points."""
    camera_pose = np.eye(4)

    points_3d = np.random.randn(8, 3)
    points_2d = np.random.randn(8, 2)

    camera = {"focal": 1.0, "principal_point": [0.5, 0.5]}

    with pytest.raises(ValueError, match="at least 10"):
        refine_camera_full(points_3d, points_2d, camera, camera_pose, (1920, 1080))


def test_refine_camera_full_custom_order():
    """Test custom optimization order."""
    true_focal = 0.9
    true_pp = (0.52, 0.48)

    camera_pose = np.eye(4)
    image_size = (1920, 1080)

    points_3d, points_2d = create_calibration_data(
        70, true_focal, true_pp, {}, camera_pose, image_size, noise_std=0.5
    )

    camera = {
        "focal": 1.0,
        "principal_point": [0.5, 0.5],
        "projection_type": "perspective",
    }

    # Custom order: PP first, then focal
    result = refine_camera_full(
        points_3d,
        points_2d,
        camera,
        camera_pose,
        image_size,
        optimize_order=["pp", "focal"],
    )

    assert result.total_iterations > 0
    assert result.improvement > 0


def test_refine_camera_full_ransac():
    """Test full refinement with RANSAC enabled."""
    true_focal = 0.85

    camera_pose = np.eye(4)
    image_size = (1920, 1080)

    points_3d, points_2d = create_calibration_data(
        100, true_focal, (0.5, 0.5), {}, camera_pose, image_size, noise_std=1.0
    )

    camera = {
        "focal": 1.0,
        "principal_point": [0.5, 0.5],
        "projection_type": "perspective",
    }

    result = refine_camera_full(
        points_3d,
        points_2d,
        camera,
        camera_pose,
        image_size,
        use_ransac=True,
        optimize_order=["focal"],
    )

    assert result.total_iterations > 0


def test_refine_camera_quick():
    """Test quick refinement (focal + PP only)."""
    true_focal = 0.88
    true_pp = (0.52, 0.48)

    camera_pose = np.eye(4)
    image_size = (1920, 1080)

    points_3d, points_2d = create_calibration_data(
        60, true_focal, true_pp, {}, camera_pose, image_size, noise_std=0.5
    )

    camera = {
        "focal": 1.0,
        "principal_point": [0.5, 0.5],  # Default PP
        "projection_type": "perspective",
    }

    result = refine_camera_quick(points_3d, points_2d, camera, camera_pose, image_size)

    assert result.method_used == "quick"
    assert result.total_iterations == 1
    assert result.improvement > 0

    # Focal should improve
    assert abs(result.refined_camera["focal"] - true_focal) < 0.2


def test_refine_camera_quick_non_default_pp():
    """Test quick refinement skips PP if not default."""
    true_focal = 0.85

    camera_pose = np.eye(4)
    image_size = (1920, 1080)

    points_3d, points_2d = create_calibration_data(
        50, true_focal, (0.51, 0.49), {}, camera_pose, image_size
    )

    camera = {
        "focal": 1.0,
        "principal_point": [0.51, 0.49],  # Non-default
        "projection_type": "perspective",
    }

    result = refine_camera_quick(points_3d, points_2d, camera, camera_pose, image_size)

    # PP should stay close to initial (not refined much)
    assert abs(result.refined_camera["principal_point"][0] - 0.51) < 0.05


def test_refine_sequence_cameras():
    """Test sequence-level refinement."""
    camera_pose = np.eye(4)
    image_size = (1920, 1080)

    # Create data for 3 images
    sequence_data = {}
    correspondences = {}
    poses = {}
    image_sizes = {}

    for i in range(3):
        image_id = f"img_{i}"

        # Each image has slightly different true parameters
        true_focal = 0.85 + i * 0.01
        points_3d, points_2d = create_calibration_data(
            50, true_focal, (0.5, 0.5), {}, camera_pose, image_size, noise_std=0.5
        )

        sequence_data[image_id] = {
            "focal": 1.0,
            "principal_point": [0.5, 0.5],
            "projection_type": "perspective",
        }
        correspondences[image_id] = (points_3d, points_2d)
        poses[image_id] = camera_pose
        image_sizes[image_id] = image_size

    results = refine_sequence_cameras(
        sequence_data, correspondences, poses, image_sizes, method="quick"
    )

    assert len(results) == 3

    # All should have improved
    for image_id, result in results.items():
        assert result.improvement > 0
        # Quick method has improvement, which means final < initial
        assert result.final_rmse >= 0


def test_refine_sequence_cameras_missing_data():
    """Test sequence refinement handles missing data gracefully."""
    sequence_data = {
        "img_0": {"focal": 1.0, "principal_point": [0.5, 0.5]},
        "img_1": {"focal": 1.0, "principal_point": [0.5, 0.5]},
    }

    # Only provide data for img_0
    correspondences = {"img_0": (np.random.randn(50, 3), np.random.randn(50, 2))}
    poses = {"img_0": np.eye(4)}
    image_sizes = {"img_0": (1920, 1080)}

    results = refine_sequence_cameras(
        sequence_data, correspondences, poses, image_sizes
    )

    # Should only have result for img_0
    assert len(results) == 1
    assert "img_0" in results


def test_self_calibration_result_dataclass():
    """Test SelfCalibrationResult dataclass."""
    original = {"focal": 1.0}
    refined = {"focal": 0.85}

    result = SelfCalibrationResult(
        original_camera=original,
        refined_camera=refined,
        validation_report={},
        refinement_history=[],
        total_iterations=3,
        converged=True,
        final_rmse=0.8,
        improvement=2.5,
        confidence=0.92,
        method_used="full_iterative",
    )

    assert result.original_camera == original
    assert result.refined_camera == refined
    assert result.total_iterations == 3
    assert result.converged
    assert result.final_rmse == 0.8
    assert result.improvement == 2.5
    assert result.confidence == 0.92
    assert result.method_used == "full_iterative"


def test_iteration_result_dataclass():
    """Test IterationResult dataclass."""
    result = IterationResult(
        iteration=2,
        focal=0.88,
        principal_point=(0.51, 0.49),
        distortion_coeffs={"k1": -0.15},
        rmse=1.2,
        improvement=0.5,
        converged=False,
    )

    assert result.iteration == 2
    assert result.focal == 0.88
    assert result.principal_point == (0.51, 0.49)
    assert result.distortion_coeffs == {"k1": -0.15}
    assert result.rmse == 1.2
    assert result.improvement == 0.5
    assert not result.converged


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
