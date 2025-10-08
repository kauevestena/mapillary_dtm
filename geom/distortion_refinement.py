"""Distortion coefficient refinement for camera intrinsics.

This module implements refinement strategies for radial and tangential distortion:
1. Brown-Conrady model (perspective/wide-angle cameras): k1, k2, k3, p1, p2
2. Fisheye model (equidistant projection): k1, k2, k3, k4
3. Levenberg-Marquardt optimization with proper regularization
4. Iterative refinement with focal length coupling

These methods improve reconstruction accuracy especially for:
- Fisheye/spherical Mapillary imagery
- Wide-angle car dashboard cameras
- Correcting manufacturer defaults
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal

import numpy as np
from scipy.optimize import least_squares

logger = logging.getLogger(__name__)


@dataclass
class DistortionRefinementResult:
    """Result from distortion coefficient refinement."""

    original_coeffs: Dict[str, float]
    refined_coeffs: Dict[str, float]
    improvement: float  # Reduction in reprojection error (pixels)
    iterations: int
    converged: bool
    model_type: Literal["brown", "fisheye", "none"]
    confidence: float  # 0-1 score based on error reduction
    final_rmse: float  # Final RMSE in pixels


def refine_distortion_brown(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    focal: float,
    camera_pose: np.ndarray,
    principal_point: Tuple[float, float],
    image_size: Tuple[int, int],
    initial_coeffs: Optional[Dict[str, float]] = None,
    optimize_focal: bool = False,
    regularization: float = 0.01,
) -> DistortionRefinementResult:
    """Refine Brown-Conrady distortion coefficients (k1, k2, k3, p1, p2).

    The Brown-Conrady model handles radial and tangential distortion:
        x_d = x * (1 + k1*r² + k2*r⁴ + k3*r⁶) + 2*p1*x*y + p2*(r² + 2*x²)
        y_d = y * (1 + k1*r² + k2*r⁴ + k3*r⁶) + p1*(r² + 2*y²) + 2*p2*x*y

    Uses Levenberg-Marquardt with Tikhonov regularization to prevent overfitting.

    Args:
        points_3d: Nx3 array of 3D world points
        points_2d: Nx2 array of corresponding 2D image points (normalized coords)
        focal: Focal length (normalized by width)
        camera_pose: 4x4 camera-to-world transformation matrix
        principal_point: (cx, cy) principal point in normalized coordinates
        image_size: (width, height) of image in pixels
        initial_coeffs: Initial distortion coefficients (or None for zero initialization)
        optimize_focal: Whether to jointly optimize focal length (default: False)
        regularization: L2 regularization weight (default: 0.01)

    Returns:
        DistortionRefinementResult with refined coefficients and quality metrics

    Raises:
        ValueError: If insufficient correspondences (need >= 10 points)
    """
    if len(points_3d) < 10:
        raise ValueError(
            f"Need at least 10 point correspondences for distortion, got {len(points_3d)}"
        )

    logger.info(
        f"Refining Brown-Conrady distortion from {len(points_3d)} correspondences"
    )

    # Initialize coefficients
    if initial_coeffs is None:
        initial_coeffs = {"k1": 0.0, "k2": 0.0, "k3": 0.0, "p1": 0.0, "p2": 0.0}

    # Convert pose to camera coordinates
    world_to_cam = np.linalg.inv(camera_pose)
    R = world_to_cam[:3, :3]
    t = world_to_cam[:3, 3]
    points_cam = (R @ points_3d.T).T + t

    # Filter points behind camera
    valid_mask = points_cam[:, 2] > 0
    if valid_mask.sum() < 10:
        raise ValueError(f"Only {valid_mask.sum()} points in front of camera")

    points_cam = points_cam[valid_mask]
    points_2d = points_2d[valid_mask]

    # Convert 2D points to pixels for residual computation
    width, height = image_size
    obs_px = points_2d[:, 0] * width
    obs_py = points_2d[:, 1] * height

    # Parameter vector: [k1, k2, k3, p1, p2] or [k1, k2, k3, p1, p2, focal]
    if optimize_focal:
        x0 = np.array(
            [
                initial_coeffs.get("k1", 0.0),
                initial_coeffs.get("k2", 0.0),
                initial_coeffs.get("k3", 0.0),
                initial_coeffs.get("p1", 0.0),
                initial_coeffs.get("p2", 0.0),
                focal,
            ]
        )
    else:
        x0 = np.array(
            [
                initial_coeffs.get("k1", 0.0),
                initial_coeffs.get("k2", 0.0),
                initial_coeffs.get("k3", 0.0),
                initial_coeffs.get("p1", 0.0),
                initial_coeffs.get("p2", 0.0),
            ]
        )

    def residuals(params: np.ndarray) -> np.ndarray:
        """Compute reprojection residuals with regularization."""
        if optimize_focal:
            k1, k2, k3, p1, p2, f = params
        else:
            k1, k2, k3, p1, p2 = params
            f = focal

        # Project to normalized image plane
        proj_x = points_cam[:, 0] / points_cam[:, 2]
        proj_y = points_cam[:, 1] / points_cam[:, 2]

        # Apply Brown-Conrady distortion
        r2 = proj_x**2 + proj_y**2
        radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3

        x_dist = proj_x * radial + 2 * p1 * proj_x * proj_y + p2 * (r2 + 2 * proj_x**2)
        y_dist = proj_y * radial + p1 * (r2 + 2 * proj_y**2) + 2 * p2 * proj_x * proj_y

        # Convert to pixels
        px = f * width * x_dist + principal_point[0] * width
        py = f * width * y_dist + principal_point[1] * height

        # Compute residuals (x and y separately for Jacobian)
        res_x = px - obs_px
        res_y = py - obs_py

        residuals_vec = np.concatenate([res_x, res_y])

        # Add L2 regularization to prevent extreme coefficients
        reg_term = np.sqrt(regularization) * params[:5]  # Don't regularize focal

        return np.concatenate([residuals_vec, reg_term])

    # Compute initial error
    initial_residuals = residuals(x0)
    # Exclude regularization term from RMSE
    n_points = len(points_cam)
    initial_rmse = np.sqrt(np.mean(initial_residuals[: 2 * n_points] ** 2))

    # Optimize with Levenberg-Marquardt
    result = least_squares(
        residuals, x0, method="lm", max_nfev=200, ftol=1e-8, xtol=1e-8
    )

    # Extract refined parameters
    if optimize_focal:
        k1, k2, k3, p1, p2, refined_focal = result.x
    else:
        k1, k2, k3, p1, p2 = result.x
        refined_focal = focal

    refined_coeffs = {"k1": k1, "k2": k2, "k3": k3, "p1": p1, "p2": p2}

    if optimize_focal:
        refined_coeffs["focal"] = refined_focal

    # Compute final error
    final_residuals = result.fun
    final_rmse = np.sqrt(np.mean(final_residuals[: 2 * n_points] ** 2))

    improvement = initial_rmse - final_rmse

    # Compute confidence based on error reduction
    confidence = np.clip(improvement / (initial_rmse + 1e-6), 0, 1)

    # Penalize if coefficients are extreme
    if abs(k1) > 1.0 or abs(k2) > 0.5 or abs(k3) > 0.3:
        confidence *= 0.7
        logger.warning(
            f"Extreme distortion coefficients: k1={k1:.4f}, k2={k2:.4f}, k3={k3:.4f}"
        )

    # Penalize if final error is still high
    if final_rmse > 2.0:
        confidence *= 0.5

    logger.info(
        f"Brown-Conrady refined: RMSE {initial_rmse:.2f} -> {final_rmse:.2f} px, "
        f"k1={k1:.4f}, k2={k2:.4f}, p1={p1:.4f}"
    )

    return DistortionRefinementResult(
        original_coeffs=initial_coeffs,
        refined_coeffs=refined_coeffs,
        improvement=improvement,
        iterations=result.nfev,
        converged=result.success,
        model_type="brown",
        confidence=confidence,
        final_rmse=final_rmse,
    )


def refine_distortion_fisheye(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    focal: float,
    camera_pose: np.ndarray,
    principal_point: Tuple[float, float],
    image_size: Tuple[int, int],
    initial_coeffs: Optional[Dict[str, float]] = None,
    optimize_focal: bool = False,
    regularization: float = 0.01,
) -> DistortionRefinementResult:
    """Refine fisheye distortion coefficients (k1, k2, k3, k4).

    The fisheye model uses equidistant projection:
        r_d = θ * (1 + k1*θ² + k2*θ⁴ + k3*θ⁶ + k4*θ⁸)

    where θ is the angle from the optical axis.

    This model is appropriate for:
    - Wide-angle fisheye cameras (FOV > 120°)
    - Mapillary spherical imagery
    - GoPro-style action cameras

    Args:
        points_3d: Nx3 array of 3D world points
        points_2d: Nx2 array of corresponding 2D image points (normalized coords)
        focal: Focal length (normalized by width)
        camera_pose: 4x4 camera-to-world transformation matrix
        principal_point: (cx, cy) principal point in normalized coordinates
        image_size: (width, height) of image in pixels
        initial_coeffs: Initial distortion coefficients (or None for zero initialization)
        optimize_focal: Whether to jointly optimize focal length (default: False)
        regularization: L2 regularization weight (default: 0.01)

    Returns:
        DistortionRefinementResult with refined fisheye coefficients

    Raises:
        ValueError: If insufficient correspondences (need >= 10 points)
    """
    if len(points_3d) < 10:
        raise ValueError(
            f"Need at least 10 point correspondences for distortion, got {len(points_3d)}"
        )

    logger.info(f"Refining fisheye distortion from {len(points_3d)} correspondences")

    # Initialize coefficients
    if initial_coeffs is None:
        initial_coeffs = {"k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0}

    # Convert pose to camera coordinates
    world_to_cam = np.linalg.inv(camera_pose)
    R = world_to_cam[:3, :3]
    t = world_to_cam[:3, 3]
    points_cam = (R @ points_3d.T).T + t

    # Filter points behind camera
    valid_mask = points_cam[:, 2] > 0
    if valid_mask.sum() < 10:
        raise ValueError(f"Only {valid_mask.sum()} points in front of camera")

    points_cam = points_cam[valid_mask]
    points_2d = points_2d[valid_mask]

    # Convert 2D points to pixels
    width, height = image_size
    obs_px = points_2d[:, 0] * width
    obs_py = points_2d[:, 1] * height

    # Parameter vector
    if optimize_focal:
        x0 = np.array(
            [
                initial_coeffs.get("k1", 0.0),
                initial_coeffs.get("k2", 0.0),
                initial_coeffs.get("k3", 0.0),
                initial_coeffs.get("k4", 0.0),
                focal,
            ]
        )
    else:
        x0 = np.array(
            [
                initial_coeffs.get("k1", 0.0),
                initial_coeffs.get("k2", 0.0),
                initial_coeffs.get("k3", 0.0),
                initial_coeffs.get("k4", 0.0),
            ]
        )

    def residuals(params: np.ndarray) -> np.ndarray:
        """Compute reprojection residuals with fisheye model."""
        if optimize_focal:
            k1, k2, k3, k4, f = params
        else:
            k1, k2, k3, k4 = params
            f = focal

        # Compute angle from optical axis
        r = np.sqrt(points_cam[:, 0] ** 2 + points_cam[:, 1] ** 2)
        theta = np.arctan2(r, points_cam[:, 2])

        # Apply fisheye distortion
        theta2 = theta**2
        r_d = theta * (
            1 + k1 * theta2 + k2 * theta2**2 + k3 * theta2**3 + k4 * theta2**4
        )

        # Compute distorted coordinates
        scale = r_d / (r + 1e-10)  # Avoid division by zero
        x_dist = points_cam[:, 0] * scale
        y_dist = points_cam[:, 1] * scale

        # Convert to pixels
        px = f * width * x_dist + principal_point[0] * width
        py = f * width * y_dist + principal_point[1] * height

        # Compute residuals
        res_x = px - obs_px
        res_y = py - obs_py

        residuals_vec = np.concatenate([res_x, res_y])

        # Add L2 regularization
        reg_term = np.sqrt(regularization) * params[:4]  # Don't regularize focal

        return np.concatenate([residuals_vec, reg_term])

    # Compute initial error
    initial_residuals = residuals(x0)
    n_points = len(points_cam)
    initial_rmse = np.sqrt(np.mean(initial_residuals[: 2 * n_points] ** 2))

    # Optimize with Levenberg-Marquardt
    result = least_squares(
        residuals, x0, method="lm", max_nfev=200, ftol=1e-8, xtol=1e-8
    )

    # Extract refined parameters
    if optimize_focal:
        k1, k2, k3, k4, refined_focal = result.x
    else:
        k1, k2, k3, k4 = result.x
        refined_focal = focal

    refined_coeffs = {"k1": k1, "k2": k2, "k3": k3, "k4": k4}

    if optimize_focal:
        refined_coeffs["focal"] = refined_focal

    # Compute final error
    final_residuals = result.fun
    final_rmse = np.sqrt(np.mean(final_residuals[: 2 * n_points] ** 2))

    improvement = initial_rmse - final_rmse
    confidence = np.clip(improvement / (initial_rmse + 1e-6), 0, 1)

    # Penalize extreme coefficients
    if abs(k1) > 1.5 or abs(k2) > 1.0:
        confidence *= 0.7
        logger.warning(f"Extreme fisheye coefficients: k1={k1:.4f}, k2={k2:.4f}")

    if final_rmse > 2.0:
        confidence *= 0.5

    logger.info(
        f"Fisheye refined: RMSE {initial_rmse:.2f} -> {final_rmse:.2f} px, "
        f"k1={k1:.4f}, k2={k2:.4f}"
    )

    return DistortionRefinementResult(
        original_coeffs=initial_coeffs,
        refined_coeffs=refined_coeffs,
        improvement=improvement,
        iterations=result.nfev,
        converged=result.success,
        model_type="fisheye",
        confidence=confidence,
        final_rmse=final_rmse,
    )


def refine_distortion_auto(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    focal: float,
    camera_pose: np.ndarray,
    principal_point: Tuple[float, float],
    image_size: Tuple[int, int],
    projection_type: str,
    initial_coeffs: Optional[Dict[str, float]] = None,
    optimize_focal: bool = False,
) -> DistortionRefinementResult:
    """Automatically select and refine appropriate distortion model.

    Chooses between Brown-Conrady and fisheye based on projection type:
    - 'perspective', 'brown': Brown-Conrady model
    - 'fisheye', 'equidistant': Fisheye model
    - 'spherical': No distortion (return identity)

    Args:
        points_3d: Nx3 array of 3D world points
        points_2d: Nx2 array of corresponding 2D image points
        focal: Focal length (normalized by width)
        camera_pose: 4x4 camera-to-world transformation matrix
        principal_point: (cx, cy) principal point in normalized coordinates
        image_size: (width, height) of image in pixels
        projection_type: Camera projection type string
        initial_coeffs: Initial distortion coefficients
        optimize_focal: Whether to jointly optimize focal length

    Returns:
        DistortionRefinementResult with appropriate model
    """
    projection_lower = projection_type.lower()

    if projection_lower in ["spherical", "equirectangular"]:
        # Spherical cameras don't need distortion
        logger.info("Spherical camera detected - skipping distortion refinement")
        return DistortionRefinementResult(
            original_coeffs={},
            refined_coeffs={},
            improvement=0.0,
            iterations=0,
            converged=True,
            model_type="none",
            confidence=1.0,
            final_rmse=0.0,
        )

    elif projection_lower in ["fisheye", "equidistant"]:
        return refine_distortion_fisheye(
            points_3d,
            points_2d,
            focal,
            camera_pose,
            principal_point,
            image_size,
            initial_coeffs,
            optimize_focal,
        )

    else:
        # Default to Brown-Conrady for perspective/wide-angle
        return refine_distortion_brown(
            points_3d,
            points_2d,
            focal,
            camera_pose,
            principal_point,
            image_size,
            initial_coeffs,
            optimize_focal,
        )


def iterative_refinement(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    initial_focal: float,
    camera_pose: np.ndarray,
    principal_point: Tuple[float, float],
    image_size: Tuple[int, int],
    projection_type: str,
    initial_coeffs: Optional[Dict[str, float]] = None,
    max_iterations: int = 3,
    convergence_threshold: float = 0.1,
) -> Tuple[float, Dict[str, float], DistortionRefinementResult]:
    """Iteratively refine focal length and distortion together.

    This implements a coordinate descent approach:
    1. Fix distortion, optimize focal
    2. Fix focal, optimize distortion
    3. Repeat until convergence

    This is more robust than joint optimization when initial parameters are poor.

    Args:
        points_3d: Nx3 array of 3D world points
        points_2d: Nx2 array of corresponding 2D image points
        initial_focal: Starting focal length
        camera_pose: 4x4 camera-to-world transformation matrix
        principal_point: (cx, cy) principal point
        image_size: (width, height) in pixels
        projection_type: Camera projection type
        initial_coeffs: Initial distortion coefficients
        max_iterations: Maximum refinement iterations (default: 3)
        convergence_threshold: RMSE change threshold for convergence (pixels)

    Returns:
        Tuple of (refined_focal, refined_distortion_coeffs, final_result)
    """
    from .focal_refinement import refine_focal_geometric

    logger.info(f"Starting iterative refinement (max {max_iterations} iterations)")

    current_focal = initial_focal
    current_coeffs = initial_coeffs or {}
    prev_rmse = float("inf")

    for iteration in range(max_iterations):
        # Step 1: Refine focal with fixed distortion
        focal_result = refine_focal_geometric(
            points_3d,
            points_2d,
            current_focal,
            camera_pose,
            principal_point,
            image_size,
            current_coeffs,
        )
        current_focal = focal_result.refined_focal

        # Step 2: Refine distortion with fixed focal
        dist_result = refine_distortion_auto(
            points_3d,
            points_2d,
            current_focal,
            camera_pose,
            principal_point,
            image_size,
            projection_type,
            current_coeffs,
            optimize_focal=False,
        )
        current_coeffs = dist_result.refined_coeffs

        # Check convergence
        rmse_change = abs(prev_rmse - dist_result.final_rmse)
        logger.info(
            f"Iteration {iteration+1}: focal={current_focal:.4f}, "
            f"RMSE={dist_result.final_rmse:.2f} px (change: {rmse_change:.2f})"
        )

        if rmse_change < convergence_threshold:
            logger.info(f"Converged after {iteration+1} iterations")
            break

        prev_rmse = dist_result.final_rmse

    return current_focal, current_coeffs, dist_result
