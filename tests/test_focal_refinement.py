"""Tests for focal length refinement algorithms."""
from __future__ import annotations

import pytest
import numpy as np

from geom.focal_refinement import (
    FocalRefinementResult,
    refine_focal_geometric,
    refine_focal_ransac,
    apply_distortion,
    refine_focal_bundle_adjustment,
)


def create_synthetic_correspondences(
    n_points: int,
    true_focal: float,
    camera_pose: np.ndarray,
    principal_point: tuple = (0.5, 0.5),
    image_size: tuple = (1920, 1080),
    noise_std: float = 0.0,
    outlier_ratio: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic 3D-2D point correspondences for testing.
    
    Args:
        n_points: Number of points to generate
        true_focal: Ground truth focal length
        camera_pose: 4x4 camera-to-world transformation
        principal_point: (cx, cy) in normalized coords
        image_size: (width, height) in pixels
        noise_std: Gaussian noise std deviation in pixels
        outlier_ratio: Fraction of points to make outliers (0-1)
        
    Returns:
        Tuple of (points_3d, points_2d_normalized)
    """
    # Generate random 3D points in front of camera
    rng = np.random.RandomState(42)
    points_3d_cam = np.zeros((n_points, 3))
    points_3d_cam[:, 0] = rng.uniform(-5, 5, n_points)  # X: -5 to 5 meters
    points_3d_cam[:, 1] = rng.uniform(-3, 3, n_points)  # Y: -3 to 3 meters
    points_3d_cam[:, 2] = rng.uniform(5, 20, n_points)  # Z: 5 to 20 meters
    
    # Transform to world coordinates
    points_3d = (camera_pose[:3, :3] @ points_3d_cam.T).T + camera_pose[:3, 3]
    
    # Project to image plane
    proj_x = points_3d_cam[:, 0] / points_3d_cam[:, 2]
    proj_y = points_3d_cam[:, 1] / points_3d_cam[:, 2]
    
    width, height = image_size
    px = true_focal * width * proj_x + principal_point[0] * width
    py = true_focal * width * proj_y + principal_point[1] * height
    
    # Add noise
    if noise_std > 0:
        px += rng.normal(0, noise_std, n_points)
        py += rng.normal(0, noise_std, n_points)
    
    # Add outliers
    if outlier_ratio > 0:
        n_outliers = int(n_points * outlier_ratio)
        outlier_indices = rng.choice(n_points, n_outliers, replace=False)
        px[outlier_indices] += rng.uniform(-50, 50, n_outliers)
        py[outlier_indices] += rng.uniform(-50, 50, n_outliers)
    
    # Convert to normalized coordinates
    points_2d = np.column_stack([px / width, py / height])
    
    return points_3d, points_2d


def test_refine_focal_geometric_perfect_data():
    """Test geometric refinement with perfect synthetic data."""
    true_focal = 0.85
    wrong_focal = 1.0  # Start with default
    
    # Create camera pose (identity = camera at origin)
    camera_pose = np.eye(4)
    
    # Generate perfect correspondences
    points_3d, points_2d = create_synthetic_correspondences(
        n_points=50,
        true_focal=true_focal,
        camera_pose=camera_pose,
        noise_std=0.0
    )
    
    result = refine_focal_geometric(
        points_3d=points_3d,
        points_2d=points_2d,
        initial_focal=wrong_focal,
        camera_pose=camera_pose,
        principal_point=(0.5, 0.5),
        image_size=(1920, 1080)
    )
    
    assert result.converged
    assert abs(result.refined_focal - true_focal) < 0.01  # Within 1%
    assert result.improvement > 0  # Error should decrease
    assert result.confidence > 0.9  # High confidence with perfect data
    assert result.method == 'geometric'


def test_refine_focal_geometric_noisy_data():
    """Test geometric refinement with noisy measurements."""
    true_focal = 0.75
    initial_focal = 0.9
    
    camera_pose = np.eye(4)
    
    # Add 1 pixel noise
    points_3d, points_2d = create_synthetic_correspondences(
        n_points=100,
        true_focal=true_focal,
        camera_pose=camera_pose,
        noise_std=1.0
    )
    
    result = refine_focal_geometric(
        points_3d=points_3d,
        points_2d=points_2d,
        initial_focal=initial_focal,
        camera_pose=camera_pose,
        principal_point=(0.5, 0.5),
        image_size=(1920, 1080)
    )
    
    assert result.converged
    # With noise, allow 5% tolerance
    assert abs(result.refined_focal - true_focal) < 0.05
    assert result.improvement > 0


def test_refine_focal_geometric_with_distortion():
    """Test geometric refinement with distortion coefficients."""
    true_focal = 0.8
    initial_focal = 0.95
    
    camera_pose = np.eye(4)
    distortion = {'k1': -0.2, 'k2': 0.05, 'p1': 0.001, 'p2': -0.001}
    
    points_3d, points_2d = create_synthetic_correspondences(
        n_points=80,
        true_focal=true_focal,
        camera_pose=camera_pose,
        noise_std=0.5
    )
    
    result = refine_focal_geometric(
        points_3d=points_3d,
        points_2d=points_2d,
        initial_focal=initial_focal,
        camera_pose=camera_pose,
        principal_point=(0.5, 0.5),
        image_size=(1920, 1080),
        distortion_coeffs=distortion
    )
    
    assert result.converged
    assert result.improvement > 0


def test_refine_focal_geometric_insufficient_points():
    """Test that geometric refinement raises error with too few points."""
    camera_pose = np.eye(4)
    
    points_3d = np.random.randn(4, 3)  # Only 4 points
    points_2d = np.random.randn(4, 2)
    
    with pytest.raises(ValueError, match="at least 6"):
        refine_focal_geometric(
            points_3d=points_3d,
            points_2d=points_2d,
            initial_focal=1.0,
            camera_pose=camera_pose,
            principal_point=(0.5, 0.5),
            image_size=(1920, 1080)
        )


def test_refine_focal_geometric_points_behind_camera():
    """Test handling of points behind camera."""
    camera_pose = np.eye(4)
    
    # Create more points, but all behind camera (negative Z)
    points_3d = np.array([
        [1, 1, -5],
        [2, -1, -3],
        [-1, 2, -4],
        [0.5, 0.5, -6],
        [-0.5, 1.5, -7],
        [1.5, -0.5, -8],
        [0, 0, -5]
    ])
    points_2d = np.random.randn(7, 2)
    
    with pytest.raises(ValueError, match="in front of camera"):
        refine_focal_geometric(
            points_3d=points_3d,
            points_2d=points_2d,
            initial_focal=1.0,
            camera_pose=camera_pose,
            principal_point=(0.5, 0.5),
            image_size=(1920, 1080)
        )


def test_refine_focal_ransac_with_outliers():
    """Test RANSAC refinement with outlier correspondences."""
    true_focal = 0.82
    initial_focal = 1.0
    
    camera_pose = np.eye(4)
    
    # Generate data with 30% outliers
    points_3d, points_2d = create_synthetic_correspondences(
        n_points=100,
        true_focal=true_focal,
        camera_pose=camera_pose,
        noise_std=0.5,
        outlier_ratio=0.3
    )
    
    result = refine_focal_ransac(
        points_3d=points_3d,
        points_2d=points_2d,
        initial_focal=initial_focal,
        camera_pose=camera_pose,
        principal_point=(0.5, 0.5),
        image_size=(1920, 1080),
        ransac_threshold=2.0,
        ransac_iterations=100
    )
    
    # Should find reasonable inlier set (convergence may vary with random sampling)
    assert result.method == 'ransac'
    # RANSAC should handle outliers well - check focal is reasonable
    assert 0.5 < result.refined_focal < 1.5
    assert abs(result.refined_focal - true_focal) < 0.15  # Within 15% with outliers


def test_refine_focal_ransac_clean_data():
    """Test RANSAC refinement with clean data (no outliers)."""
    true_focal = 0.88
    initial_focal = 1.05
    
    camera_pose = np.eye(4)
    
    points_3d, points_2d = create_synthetic_correspondences(
        n_points=50,
        true_focal=true_focal,
        camera_pose=camera_pose,
        noise_std=0.5,
        outlier_ratio=0.0
    )
    
    result = refine_focal_ransac(
        points_3d=points_3d,
        points_2d=points_2d,
        initial_focal=initial_focal,
        camera_pose=camera_pose,
        principal_point=(0.5, 0.5),
        image_size=(1920, 1080)
    )
    
    assert result.converged
    assert result.confidence > 0.7  # High inlier ratio
    assert abs(result.refined_focal - true_focal) < 0.1


def test_refine_focal_ransac_insufficient_points():
    """Test RANSAC with insufficient points."""
    camera_pose = np.eye(4)
    
    points_3d = np.random.randn(8, 3)
    points_2d = np.random.randn(8, 2)
    
    with pytest.raises(ValueError, match="at least 10"):
        refine_focal_ransac(
            points_3d=points_3d,
            points_2d=points_2d,
            initial_focal=1.0,
            camera_pose=camera_pose,
            principal_point=(0.5, 0.5),
            image_size=(1920, 1080)
        )


def test_apply_distortion_brown_conrady():
    """Test Brown-Conrady distortion model."""
    # Test points
    x = np.array([0.0, 0.1, -0.1, 0.2])
    y = np.array([0.0, 0.1, 0.1, -0.2])
    
    # Typical coefficients
    coeffs = {
        'k1': -0.2,
        'k2': 0.05,
        'k3': 0.01,
        'p1': 0.001,
        'p2': -0.001
    }
    
    x_dist, y_dist = apply_distortion(x, y, coeffs)
    
    # Distortion should modify points
    assert not np.allclose(x_dist, x)
    assert not np.allclose(y_dist, y)
    
    # Origin should remain at origin
    assert abs(x_dist[0]) < 1e-10
    assert abs(y_dist[0]) < 1e-10


def test_apply_distortion_no_coefficients():
    """Test distortion with zero coefficients (identity)."""
    x = np.array([0.1, 0.2, -0.1])
    y = np.array([0.1, -0.2, 0.15])
    
    coeffs = {}  # Empty dict = no distortion
    
    x_dist, y_dist = apply_distortion(x, y, coeffs)
    
    # Should be identity transformation
    np.testing.assert_allclose(x_dist, x)
    np.testing.assert_allclose(y_dist, y)


def test_apply_distortion_radial_only():
    """Test radial distortion only (no tangential)."""
    x = np.array([0.0, 0.3])
    y = np.array([0.0, 0.4])
    
    coeffs = {'k1': -0.3, 'k2': 0.1}
    
    x_dist, y_dist = apply_distortion(x, y, coeffs)
    
    # Origin unchanged
    assert abs(x_dist[0]) < 1e-10
    
    # Radial distortion should affect both x and y proportionally
    r = np.sqrt(x[1]**2 + y[1]**2)
    radial_factor = 1 + coeffs['k1']*r**2 + coeffs['k2']*r**4
    
    np.testing.assert_allclose(x_dist[1], x[1] * radial_factor, rtol=1e-5)


def test_refine_focal_bundle_adjustment_placeholder():
    """Test bundle adjustment placeholder (to be replaced with real implementation)."""
    sequences = {
        'seq1': {'focal': 0.85, 'width': 1920, 'height': 1080},
        'seq2': {'focal': 0.90, 'width': 1920, 'height': 1080},
        'seq3': {'focal': 0.80, 'width': 2048, 'height': 1536}
    }
    
    track_data = {}  # Placeholder
    
    results = refine_focal_bundle_adjustment(sequences, track_data)
    
    assert len(results) == 3
    for seq_id, result in results.items():
        assert isinstance(result, FocalRefinementResult)
        assert result.method == 'bundle_adjustment'
        assert result.converged
        # Placeholder should produce small change
        assert abs(result.refined_focal - result.original_focal) < 0.05


def test_focal_refinement_result_dataclass():
    """Test FocalRefinementResult dataclass properties."""
    result = FocalRefinementResult(
        original_focal=1.0,
        refined_focal=0.85,
        improvement=1.5,
        iterations=50,
        converged=True,
        method='geometric',
        confidence=0.92
    )
    
    assert result.original_focal == 1.0
    assert result.refined_focal == 0.85
    assert result.improvement == 1.5
    assert result.iterations == 50
    assert result.converged
    assert result.method == 'geometric'
    assert result.confidence == 0.92


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
