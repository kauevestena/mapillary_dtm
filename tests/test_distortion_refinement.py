"""Tests for distortion coefficient refinement algorithms."""

from __future__ import annotations

import pytest
import numpy as np

from geom.distortion_refinement import (
    DistortionRefinementResult,
    refine_distortion_brown,
    refine_distortion_fisheye,
    refine_distortion_auto,
    iterative_refinement,
)


def create_distorted_correspondences(
    n_points: int,
    focal: float,
    camera_pose: np.ndarray,
    distortion_coeffs: dict,
    model_type: str = "brown",
    principal_point: tuple = (0.5, 0.5),
    image_size: tuple = (1920, 1080),
    noise_std: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic correspondences with known distortion.

    Args:
        n_points: Number of points
        focal: Focal length (normalized)
        camera_pose: 4x4 camera-to-world matrix
        distortion_coeffs: True distortion coefficients
        model_type: 'brown' or 'fisheye'
        principal_point: (cx, cy) normalized
        image_size: (width, height) pixels
        noise_std: Noise standard deviation in pixels

    Returns:
        Tuple of (points_3d, points_2d_normalized)
    """
    rng = np.random.RandomState(42)

    # Generate 3D points in camera frame
    points_3d_cam = np.zeros((n_points, 3))
    points_3d_cam[:, 0] = rng.uniform(-5, 5, n_points)
    points_3d_cam[:, 1] = rng.uniform(-3, 3, n_points)
    points_3d_cam[:, 2] = rng.uniform(5, 20, n_points)

    # Transform to world
    points_3d = (camera_pose[:3, :3] @ points_3d_cam.T).T + camera_pose[:3, 3]

    # Project to normalized plane
    proj_x = points_3d_cam[:, 0] / points_3d_cam[:, 2]
    proj_y = points_3d_cam[:, 1] / points_3d_cam[:, 2]

    # Apply distortion
    if model_type == "brown":
        k1 = distortion_coeffs.get("k1", 0.0)
        k2 = distortion_coeffs.get("k2", 0.0)
        k3 = distortion_coeffs.get("k3", 0.0)
        p1 = distortion_coeffs.get("p1", 0.0)
        p2 = distortion_coeffs.get("p2", 0.0)

        r2 = proj_x**2 + proj_y**2
        radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3

        x_dist = proj_x * radial + 2 * p1 * proj_x * proj_y + p2 * (r2 + 2 * proj_x**2)
        y_dist = proj_y * radial + p1 * (r2 + 2 * proj_y**2) + 2 * p2 * proj_x * proj_y

    elif model_type == "fisheye":
        k1 = distortion_coeffs.get("k1", 0.0)
        k2 = distortion_coeffs.get("k2", 0.0)
        k3 = distortion_coeffs.get("k3", 0.0)
        k4 = distortion_coeffs.get("k4", 0.0)

        r = np.sqrt(points_3d_cam[:, 0] ** 2 + points_3d_cam[:, 1] ** 2)
        theta = np.arctan2(r, points_3d_cam[:, 2])
        theta2 = theta**2

        r_d = theta * (
            1 + k1 * theta2 + k2 * theta2**2 + k3 * theta2**3 + k4 * theta2**4
        )
        scale = r_d / (r + 1e-10)

        x_dist = points_3d_cam[:, 0] * scale
        y_dist = points_3d_cam[:, 1] * scale

    else:
        x_dist = proj_x
        y_dist = proj_y

    # Convert to pixels
    width, height = image_size
    px = focal * width * x_dist + principal_point[0] * width
    py = focal * width * y_dist + principal_point[1] * height

    # Add noise
    if noise_std > 0:
        px += rng.normal(0, noise_std, n_points)
        py += rng.normal(0, noise_std, n_points)

    # Convert to normalized coordinates
    points_2d = np.column_stack([px / width, py / height])

    return points_3d, points_2d


def test_refine_distortion_brown_perfect():
    """Test Brown-Conrady refinement with perfect synthetic data."""
    true_focal = 0.85
    true_coeffs = {"k1": -0.2, "k2": 0.05, "k3": 0.01, "p1": 0.001, "p2": -0.001}

    camera_pose = np.eye(4)

    points_3d, points_2d = create_distorted_correspondences(
        n_points=50,
        focal=true_focal,
        camera_pose=camera_pose,
        distortion_coeffs=true_coeffs,
        model_type="brown",
        noise_std=0.0,
    )

    # Start with zero distortion
    result = refine_distortion_brown(
        points_3d=points_3d,
        points_2d=points_2d,
        focal=true_focal,
        camera_pose=camera_pose,
        principal_point=(0.5, 0.5),
        image_size=(1920, 1080),
        initial_coeffs=None,
    )

    assert result.converged
    assert result.model_type == "brown"
    assert result.improvement > 0
    assert result.final_rmse < 0.5  # Very low error with perfect data

    # Check coefficients are close to truth
    assert abs(result.refined_coeffs["k1"] - true_coeffs["k1"]) < 0.05
    assert abs(result.refined_coeffs["k2"] - true_coeffs["k2"]) < 0.02


def test_refine_distortion_brown_noisy():
    """Test Brown-Conrady refinement with noisy measurements."""
    true_focal = 0.8
    true_coeffs = {"k1": -0.15, "k2": 0.03, "p1": 0.0, "p2": 0.0}

    camera_pose = np.eye(4)

    points_3d, points_2d = create_distorted_correspondences(
        n_points=100,
        focal=true_focal,
        camera_pose=camera_pose,
        distortion_coeffs=true_coeffs,
        model_type="brown",
        noise_std=0.5,
    )

    result = refine_distortion_brown(
        points_3d=points_3d,
        points_2d=points_2d,
        focal=true_focal,
        camera_pose=camera_pose,
        principal_point=(0.5, 0.5),
        image_size=(1920, 1080),
    )

    assert result.converged
    # With noise, allow larger tolerance
    assert abs(result.refined_coeffs["k1"] - true_coeffs["k1"]) < 0.1


def test_refine_distortion_brown_with_focal():
    """Test Brown-Conrady refinement jointly optimizing focal length."""
    true_focal = 0.9
    true_coeffs = {"k1": -0.25, "k2": 0.08}
    wrong_focal = 1.0

    camera_pose = np.eye(4)

    points_3d, points_2d = create_distorted_correspondences(
        n_points=80,
        focal=true_focal,
        camera_pose=camera_pose,
        distortion_coeffs=true_coeffs,
        model_type="brown",
        noise_std=0.3,
    )

    result = refine_distortion_brown(
        points_3d=points_3d,
        points_2d=points_2d,
        focal=wrong_focal,
        camera_pose=camera_pose,
        principal_point=(0.5, 0.5),
        image_size=(1920, 1080),
        optimize_focal=True,
    )

    assert result.converged
    assert "focal" in result.refined_coeffs
    # Should improve both focal and distortion
    assert abs(result.refined_coeffs["focal"] - true_focal) < 0.1


def test_refine_distortion_brown_insufficient_points():
    """Test that Brown refinement requires sufficient points."""
    camera_pose = np.eye(4)

    points_3d = np.random.randn(8, 3)  # Only 8 points
    points_2d = np.random.randn(8, 2)

    with pytest.raises(ValueError, match="at least 10"):
        refine_distortion_brown(
            points_3d=points_3d,
            points_2d=points_2d,
            focal=1.0,
            camera_pose=camera_pose,
            principal_point=(0.5, 0.5),
            image_size=(1920, 1080),
        )


def test_refine_distortion_fisheye_perfect():
    """Test fisheye refinement with perfect synthetic data."""
    true_focal = 0.65
    true_coeffs = {"k1": 0.5, "k2": -0.2, "k3": 0.05, "k4": -0.01}

    camera_pose = np.eye(4)

    points_3d, points_2d = create_distorted_correspondences(
        n_points=60,
        focal=true_focal,
        camera_pose=camera_pose,
        distortion_coeffs=true_coeffs,
        model_type="fisheye",
        noise_std=0.0,
    )

    result = refine_distortion_fisheye(
        points_3d=points_3d,
        points_2d=points_2d,
        focal=true_focal,
        camera_pose=camera_pose,
        principal_point=(0.5, 0.5),
        image_size=(1920, 1080),
        initial_coeffs=None,
    )

    assert result.converged
    assert result.model_type == "fisheye"
    assert result.improvement > 0
    assert result.final_rmse < 1.0

    # Check coefficients
    assert abs(result.refined_coeffs["k1"] - true_coeffs["k1"]) < 0.1
    assert abs(result.refined_coeffs["k2"] - true_coeffs["k2"]) < 0.05


def test_refine_distortion_fisheye_noisy():
    """Test fisheye refinement with noise."""
    true_focal = 0.7
    true_coeffs = {"k1": 0.3, "k2": -0.1}

    camera_pose = np.eye(4)

    points_3d, points_2d = create_distorted_correspondences(
        n_points=100,
        focal=true_focal,
        camera_pose=camera_pose,
        distortion_coeffs=true_coeffs,
        model_type="fisheye",
        noise_std=0.8,
    )

    result = refine_distortion_fisheye(
        points_3d=points_3d,
        points_2d=points_2d,
        focal=true_focal,
        camera_pose=camera_pose,
        principal_point=(0.5, 0.5),
        image_size=(1920, 1080),
    )

    assert result.converged
    # Noise allows larger tolerance
    assert abs(result.refined_coeffs["k1"] - true_coeffs["k1"]) < 0.15


def test_refine_distortion_fisheye_with_focal():
    """Test fisheye refinement jointly optimizing focal."""
    true_focal = 0.6
    true_coeffs = {"k1": 0.4, "k2": -0.15}
    wrong_focal = 0.8

    camera_pose = np.eye(4)

    points_3d, points_2d = create_distorted_correspondences(
        n_points=70,
        focal=true_focal,
        camera_pose=camera_pose,
        distortion_coeffs=true_coeffs,
        model_type="fisheye",
        noise_std=0.4,
    )

    result = refine_distortion_fisheye(
        points_3d=points_3d,
        points_2d=points_2d,
        focal=wrong_focal,
        camera_pose=camera_pose,
        principal_point=(0.5, 0.5),
        image_size=(1920, 1080),
        optimize_focal=True,
    )

    assert result.converged
    assert "focal" in result.refined_coeffs
    assert abs(result.refined_coeffs["focal"] - true_focal) < 0.15


def test_refine_distortion_auto_brown():
    """Test auto refinement selects Brown model."""
    true_focal = 0.85
    true_coeffs = {"k1": -0.18, "k2": 0.04}

    camera_pose = np.eye(4)

    points_3d, points_2d = create_distorted_correspondences(
        n_points=50,
        focal=true_focal,
        camera_pose=camera_pose,
        distortion_coeffs=true_coeffs,
        model_type="brown",
    )

    result = refine_distortion_auto(
        points_3d=points_3d,
        points_2d=points_2d,
        focal=true_focal,
        camera_pose=camera_pose,
        principal_point=(0.5, 0.5),
        image_size=(1920, 1080),
        projection_type="perspective",
    )

    assert result.model_type == "brown"
    assert result.converged


def test_refine_distortion_auto_fisheye():
    """Test auto refinement selects fisheye model."""
    true_focal = 0.7
    true_coeffs = {"k1": 0.3, "k2": -0.12}

    camera_pose = np.eye(4)

    points_3d, points_2d = create_distorted_correspondences(
        n_points=50,
        focal=true_focal,
        camera_pose=camera_pose,
        distortion_coeffs=true_coeffs,
        model_type="fisheye",
    )

    result = refine_distortion_auto(
        points_3d=points_3d,
        points_2d=points_2d,
        focal=true_focal,
        camera_pose=camera_pose,
        principal_point=(0.5, 0.5),
        image_size=(1920, 1080),
        projection_type="fisheye",
    )

    assert result.model_type == "fisheye"
    assert result.converged


def test_refine_distortion_auto_spherical():
    """Test auto refinement skips spherical cameras."""
    camera_pose = np.eye(4)

    points_3d = np.random.randn(50, 3)
    points_2d = np.random.randn(50, 2)

    result = refine_distortion_auto(
        points_3d=points_3d,
        points_2d=points_2d,
        focal=0.5,
        camera_pose=camera_pose,
        principal_point=(0.5, 0.5),
        image_size=(3840, 1920),
        projection_type="spherical",
    )

    assert result.model_type == "none"
    assert result.converged
    assert len(result.refined_coeffs) == 0


def test_iterative_refinement():
    """Test iterative refinement of focal and distortion."""
    true_focal = 0.88
    true_coeffs = {"k1": -0.22, "k2": 0.06}

    # Start with poor estimates
    initial_focal = 1.1
    initial_coeffs = {"k1": 0.0, "k2": 0.0}

    camera_pose = np.eye(4)

    points_3d, points_2d = create_distorted_correspondences(
        n_points=80,
        focal=true_focal,
        camera_pose=camera_pose,
        distortion_coeffs=true_coeffs,
        model_type="brown",
        noise_std=0.5,
    )

    refined_focal, refined_coeffs, final_result = iterative_refinement(
        points_3d=points_3d,
        points_2d=points_2d,
        initial_focal=initial_focal,
        camera_pose=camera_pose,
        principal_point=(0.5, 0.5),
        image_size=(1920, 1080),
        projection_type="perspective",
        initial_coeffs=initial_coeffs,
        max_iterations=3,
    )

    # Should converge to better estimates
    # With poor initial values and noise, convergence may be partial
    assert abs(refined_focal - true_focal) < 0.15
    # Distortion is harder to recover from poor initialization
    assert abs(refined_coeffs["k1"] - true_coeffs["k1"]) < 0.25
    assert final_result.converged


def test_iterative_refinement_quick_convergence():
    """Test iterative refinement converges quickly with good initial values."""
    true_focal = 0.85
    true_coeffs = {"k1": -0.15, "k2": 0.03}

    # Start close to truth
    initial_focal = 0.87
    initial_coeffs = {"k1": -0.12, "k2": 0.02}

    camera_pose = np.eye(4)

    points_3d, points_2d = create_distorted_correspondences(
        n_points=60,
        focal=true_focal,
        camera_pose=camera_pose,
        distortion_coeffs=true_coeffs,
        model_type="brown",
        noise_std=0.3,
    )

    refined_focal, refined_coeffs, final_result = iterative_refinement(
        points_3d=points_3d,
        points_2d=points_2d,
        initial_focal=initial_focal,
        camera_pose=camera_pose,
        principal_point=(0.5, 0.5),
        image_size=(1920, 1080),
        projection_type="perspective",
        initial_coeffs=initial_coeffs,
        max_iterations=3,
        convergence_threshold=0.1,
    )

    # Should converge quickly
    assert final_result.final_rmse < 1.0


def test_distortion_refinement_result_dataclass():
    """Test DistortionRefinementResult dataclass."""
    result = DistortionRefinementResult(
        original_coeffs={"k1": 0.0, "k2": 0.0},
        refined_coeffs={"k1": -0.2, "k2": 0.05},
        improvement=2.5,
        iterations=50,
        converged=True,
        model_type="brown",
        confidence=0.92,
        final_rmse=0.8,
    )

    assert result.original_coeffs == {"k1": 0.0, "k2": 0.0}
    assert result.refined_coeffs["k1"] == -0.2
    assert result.improvement == 2.5
    assert result.converged
    assert result.model_type == "brown"
    assert result.confidence == 0.92
    assert result.final_rmse == 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
