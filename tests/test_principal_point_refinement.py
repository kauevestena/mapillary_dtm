"""Tests for principal point refinement algorithms."""

from __future__ import annotations

import pytest
import numpy as np

from geom.principal_point_refinement import (
    PrincipalPointRefinementResult,
    compute_reprojection_errors,
    analyze_error_symmetry,
    refine_principal_point_grid,
    refine_principal_point_gradient,
    refine_principal_point_auto,
)


def create_pp_correspondences(
    n_points: int,
    focal: float,
    true_pp: tuple,
    camera_pose: np.ndarray,
    image_size: tuple = (1920, 1080),
    noise_std: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic correspondences with known principal point.

    Args:
        n_points: Number of points
        focal: Focal length (normalized)
        true_pp: True principal point (cx, cy) normalized
        camera_pose: 4x4 camera-to-world matrix
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

    # Project to image
    proj_x = points_3d_cam[:, 0] / points_3d_cam[:, 2]
    proj_y = points_3d_cam[:, 1] / points_3d_cam[:, 2]

    width, height = image_size
    px = focal * width * proj_x + true_pp[0] * width
    py = focal * width * proj_y + true_pp[1] * height

    # Add noise
    if noise_std > 0:
        px += rng.normal(0, noise_std, n_points)
        py += rng.normal(0, noise_std, n_points)

    # Convert to normalized coordinates
    points_2d = np.column_stack([px / width, py / height])

    return points_3d, points_2d


def test_compute_reprojection_errors():
    """Test reprojection error computation."""
    # Simple case: points in front of camera
    points_3d_cam = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 10.0], [-1.0, -1.0, 10.0]])

    # Expected projections with focal=1.0, pp=(0.5, 0.5), size=(1000, 1000)
    # Point 0: x=1/10=0.1 -> px=1000*0.1+500=600 -> norm=0.6
    # Point 1: y=1/10=0.1 -> py=1000*0.1+500=600 -> norm=0.6
    focal = 1.0
    pp = (0.5, 0.5)
    image_size = (1000, 1000)

    # Observed points (perfect projection)
    points_2d = np.array([[0.6, 0.5], [0.5, 0.6], [0.4, 0.4]])

    errors = compute_reprojection_errors(
        points_3d_cam, points_2d, focal, pp, image_size
    )

    # Should have zero errors (perfect match)
    np.testing.assert_allclose(errors, 0.0, atol=1e-10)


def test_compute_reprojection_errors_offset_pp():
    """Test reprojection errors with offset principal point."""
    points_3d_cam = np.array([[1.0, 0.0, 10.0]])

    # With pp=(0.5, 0.5): proj -> (0.6, 0.5)
    # With pp=(0.6, 0.5): proj -> (0.7, 0.5)
    focal = 1.0
    pp_offset = (0.6, 0.5)
    image_size = (1000, 1000)

    # Observed point at (0.6, 0.5) with pp=(0.5, 0.5)
    points_2d = np.array([[0.6, 0.5]])

    errors = compute_reprojection_errors(
        points_3d_cam, points_2d, focal, pp_offset, image_size
    )

    # Should have error of 100 pixels in x direction
    np.testing.assert_allclose(errors[0, 0], 100.0, rtol=1e-5)
    np.testing.assert_allclose(errors[0, 1], 0.0, atol=1e-10)


def test_analyze_error_symmetry_symmetric():
    """Test symmetry analysis with symmetric errors."""
    # Symmetric errors around zero
    errors = np.array([[1, 2], [-1, -2], [0.5, 1], [-0.5, -1], [0, 0]])

    symmetry = analyze_error_symmetry(errors)

    # Mean should be close to zero
    assert abs(symmetry["mean_x"]) < 0.1
    assert abs(symmetry["mean_y"]) < 0.1

    # Asymmetry score should be low
    assert symmetry["asymmetry_score"] < 0.2


def test_analyze_error_symmetry_asymmetric():
    """Test symmetry analysis with biased errors."""
    # All errors positive (systematic bias)
    errors = np.array([[5, 3], [6, 4], [5.5, 3.5], [5.2, 3.2], [4.8, 2.8]])

    symmetry = analyze_error_symmetry(errors)

    # Mean should be significantly positive
    assert symmetry["mean_x"] > 4.0
    assert symmetry["mean_y"] > 2.0

    # Asymmetry score should be high
    assert symmetry["asymmetry_score"] > 0.5


def test_refine_principal_point_grid_perfect():
    """Test grid refinement with perfect synthetic data."""
    true_pp = (0.52, 0.48)  # Slightly offset from center
    wrong_pp = (0.5, 0.5)  # Start at exact center
    focal = 0.85

    camera_pose = np.eye(4)

    points_3d, points_2d = create_pp_correspondences(
        n_points=50,
        focal=focal,
        true_pp=true_pp,
        camera_pose=camera_pose,
        noise_std=0.0,
    )

    result = refine_principal_point_grid(
        points_3d=points_3d,
        points_2d=points_2d,
        focal=focal,
        camera_pose=camera_pose,
        initial_pp=wrong_pp,
        image_size=(1920, 1080),
        search_radius=0.1,
        grid_steps=11,
    )

    assert result.converged
    assert result.improvement > 0
    assert result.final_rmse < 1.0  # Very low with perfect data

    # Should be close to true PP
    assert abs(result.refined_pp[0] - true_pp[0]) < 0.02
    assert abs(result.refined_pp[1] - true_pp[1]) < 0.02


def test_refine_principal_point_grid_noisy():
    """Test grid refinement with noisy measurements."""
    true_pp = (0.55, 0.47)
    wrong_pp = (0.5, 0.5)
    focal = 0.8

    camera_pose = np.eye(4)

    points_3d, points_2d = create_pp_correspondences(
        n_points=100,
        focal=focal,
        true_pp=true_pp,
        camera_pose=camera_pose,
        noise_std=0.5,
    )

    result = refine_principal_point_grid(
        points_3d=points_3d,
        points_2d=points_2d,
        focal=focal,
        camera_pose=camera_pose,
        initial_pp=wrong_pp,
        image_size=(1920, 1080),
    )

    assert result.converged
    # With noise, allow larger tolerance
    assert abs(result.refined_pp[0] - true_pp[0]) < 0.05
    assert abs(result.refined_pp[1] - true_pp[1]) < 0.05


def test_refine_principal_point_grid_insufficient_points():
    """Test grid refinement requires sufficient points."""
    camera_pose = np.eye(4)

    points_3d = np.random.randn(8, 3)
    points_2d = np.random.randn(8, 2)

    with pytest.raises(ValueError, match="at least 10"):
        refine_principal_point_grid(
            points_3d=points_3d,
            points_2d=points_2d,
            focal=1.0,
            camera_pose=camera_pose,
            initial_pp=(0.5, 0.5),
            image_size=(1920, 1080),
        )


def test_refine_principal_point_gradient_perfect():
    """Test gradient refinement with perfect data."""
    true_pp = (0.51, 0.49)
    wrong_pp = (0.5, 0.5)
    focal = 0.9

    camera_pose = np.eye(4)

    points_3d, points_2d = create_pp_correspondences(
        n_points=60,
        focal=focal,
        true_pp=true_pp,
        camera_pose=camera_pose,
        noise_std=0.0,
    )

    result = refine_principal_point_gradient(
        points_3d=points_3d,
        points_2d=points_2d,
        focal=focal,
        camera_pose=camera_pose,
        initial_pp=wrong_pp,
        image_size=(1920, 1080),
    )

    assert result.converged
    assert result.improvement > 0
    assert result.final_rmse < 1.0

    # Should converge to true PP
    assert abs(result.refined_pp[0] - true_pp[0]) < 0.02
    assert abs(result.refined_pp[1] - true_pp[1]) < 0.02


def test_refine_principal_point_gradient_noisy():
    """Test gradient refinement with noise."""
    true_pp = (0.53, 0.46)
    wrong_pp = (0.5, 0.5)
    focal = 0.85

    camera_pose = np.eye(4)

    points_3d, points_2d = create_pp_correspondences(
        n_points=80,
        focal=focal,
        true_pp=true_pp,
        camera_pose=camera_pose,
        noise_std=0.8,
    )

    result = refine_principal_point_gradient(
        points_3d=points_3d,
        points_2d=points_2d,
        focal=focal,
        camera_pose=camera_pose,
        initial_pp=wrong_pp,
        image_size=(1920, 1080),
    )

    assert result.converged
    # Noise allows larger tolerance
    assert abs(result.refined_pp[0] - true_pp[0]) < 0.08


def test_refine_principal_point_gradient_insufficient_points():
    """Test gradient refinement requires sufficient points."""
    camera_pose = np.eye(4)

    points_3d = np.random.randn(7, 3)
    points_2d = np.random.randn(7, 2)

    with pytest.raises(ValueError, match="at least 10"):
        refine_principal_point_gradient(
            points_3d=points_3d,
            points_2d=points_2d,
            focal=1.0,
            camera_pose=camera_pose,
            initial_pp=(0.5, 0.5),
            image_size=(1920, 1080),
        )


def test_refine_principal_point_auto_default():
    """Test auto method selects grid for default PP."""
    true_pp = (0.54, 0.48)
    default_pp = (0.5, 0.5)  # Exact default
    focal = 0.88

    camera_pose = np.eye(4)

    points_3d, points_2d = create_pp_correspondences(
        n_points=50,
        focal=focal,
        true_pp=true_pp,
        camera_pose=camera_pose,
        noise_std=0.3,
    )

    # Auto should select grid for exact default
    result = refine_principal_point_auto(
        points_3d=points_3d,
        points_2d=points_2d,
        focal=focal,
        camera_pose=camera_pose,
        initial_pp=default_pp,
        image_size=(1920, 1080),
        method="auto",
    )

    assert result.converged
    assert result.improvement > 0


def test_refine_principal_point_auto_non_default():
    """Test auto method selects gradient for non-default PP."""
    true_pp = (0.51, 0.49)
    non_default_pp = (0.52, 0.48)  # Close but not exact default
    focal = 0.82

    camera_pose = np.eye(4)

    points_3d, points_2d = create_pp_correspondences(
        n_points=50,
        focal=focal,
        true_pp=true_pp,
        camera_pose=camera_pose,
        noise_std=0.3,
    )

    # Auto should select gradient for non-default
    result = refine_principal_point_auto(
        points_3d=points_3d,
        points_2d=points_2d,
        focal=focal,
        camera_pose=camera_pose,
        initial_pp=non_default_pp,
        image_size=(1920, 1080),
        method="auto",
    )

    assert result.converged


def test_refine_principal_point_auto_explicit_grid():
    """Test explicit grid method selection."""
    true_pp = (0.52, 0.48)
    initial_pp = (0.5, 0.5)
    focal = 0.85

    camera_pose = np.eye(4)

    points_3d, points_2d = create_pp_correspondences(
        n_points=50, focal=focal, true_pp=true_pp, camera_pose=camera_pose
    )

    result = refine_principal_point_auto(
        points_3d=points_3d,
        points_2d=points_2d,
        focal=focal,
        camera_pose=camera_pose,
        initial_pp=initial_pp,
        image_size=(1920, 1080),
        method="grid",
    )

    assert result.converged


def test_refine_principal_point_auto_explicit_gradient():
    """Test explicit gradient method selection."""
    true_pp = (0.51, 0.49)
    initial_pp = (0.5, 0.5)
    focal = 0.9

    camera_pose = np.eye(4)

    points_3d, points_2d = create_pp_correspondences(
        n_points=50, focal=focal, true_pp=true_pp, camera_pose=camera_pose
    )

    result = refine_principal_point_auto(
        points_3d=points_3d,
        points_2d=points_2d,
        focal=focal,
        camera_pose=camera_pose,
        initial_pp=initial_pp,
        image_size=(1920, 1080),
        method="gradient",
    )

    assert result.converged


def test_principal_point_refinement_result_dataclass():
    """Test PrincipalPointRefinementResult dataclass."""
    result = PrincipalPointRefinementResult(
        original_pp=(0.5, 0.5),
        refined_pp=(0.52, 0.48),
        improvement=1.5,
        iterations=50,
        converged=True,
        confidence=0.85,
        final_rmse=0.9,
        asymmetry_reduction=0.12,
    )

    assert result.original_pp == (0.5, 0.5)
    assert result.refined_pp == (0.52, 0.48)
    assert result.improvement == 1.5
    assert result.iterations == 50
    assert result.converged
    assert result.confidence == 0.85
    assert result.final_rmse == 0.9
    assert result.asymmetry_reduction == 0.12


def test_refine_principal_point_clamping():
    """Test that principal point is clamped to reasonable range."""
    # This tests boundary conditions
    true_pp = (0.5, 0.5)
    focal = 0.85

    camera_pose = np.eye(4)

    points_3d, points_2d = create_pp_correspondences(
        n_points=50, focal=focal, true_pp=true_pp, camera_pose=camera_pose
    )

    # Even with large search radius, should stay within bounds
    result = refine_principal_point_grid(
        points_3d=points_3d,
        points_2d=points_2d,
        focal=focal,
        camera_pose=camera_pose,
        initial_pp=(0.5, 0.5),
        image_size=(1920, 1080),
        search_radius=0.5,  # Very large radius
    )

    # Should be clamped to [0.2, 0.8] range
    assert 0.2 <= result.refined_pp[0] <= 0.8
    assert 0.2 <= result.refined_pp[1] <= 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
