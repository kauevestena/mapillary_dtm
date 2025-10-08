"""Principal point refinement for camera intrinsics.

This module implements principal point (optical center) refinement:
1. 2D grid search to minimize asymmetric reprojection errors
2. Gradient-based optimization for fine-tuning
3. Symmetry analysis to detect off-center principal point
4. Validation against typical ranges (within ±20% of image center)

Principal point refinement is important for:
- Correcting manufacturer defaults (often set to exact image center)
- Handling optical misalignment in camera assemblies
- Improving accuracy for wide-angle and fisheye lenses
- Reducing systematic bias in 3D reconstruction
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class PrincipalPointRefinementResult:
    """Result from principal point refinement."""

    original_pp: Tuple[float, float]  # (cx, cy) in normalized coords
    refined_pp: Tuple[float, float]
    improvement: float  # Reduction in reprojection error (pixels)
    iterations: int
    converged: bool
    confidence: float  # 0-1 score
    final_rmse: float  # Final RMSE in pixels
    asymmetry_reduction: float  # Improvement in error symmetry (0-1)


def compute_reprojection_errors(
    points_3d_cam: np.ndarray,
    points_2d: np.ndarray,
    focal: float,
    principal_point: Tuple[float, float],
    image_size: Tuple[int, int],
    distortion_coeffs: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Compute per-point reprojection errors.

    Args:
        points_3d_cam: Nx3 array of 3D points in camera frame
        points_2d: Nx2 array of observed 2D points (normalized coords)
        focal: Focal length (normalized by width)
        principal_point: (cx, cy) in normalized coordinates
        image_size: (width, height) in pixels
        distortion_coeffs: Optional distortion parameters

    Returns:
        Nx2 array of reprojection errors [error_x, error_y] in pixels
    """
    # Project to normalized image plane
    proj_x = points_3d_cam[:, 0] / points_3d_cam[:, 2]
    proj_y = points_3d_cam[:, 1] / points_3d_cam[:, 2]

    # Apply distortion if provided
    if distortion_coeffs:
        from .focal_refinement import apply_distortion

        proj_x, proj_y = apply_distortion(proj_x, proj_y, distortion_coeffs)

    # Convert to pixels
    width, height = image_size
    px = focal * width * proj_x + principal_point[0] * width
    py = focal * width * proj_y + principal_point[1] * height

    # Observed points in pixels
    obs_px = points_2d[:, 0] * width
    obs_py = points_2d[:, 1] * height

    # Compute errors
    error_x = px - obs_px
    error_y = py - obs_py

    return np.column_stack([error_x, error_y])


def analyze_error_symmetry(errors: np.ndarray) -> Dict[str, float]:
    """Analyze symmetry of reprojection errors.

    A well-calibrated camera should have symmetric error distribution.
    Asymmetry indicates incorrect principal point.

    Args:
        errors: Nx2 array of reprojection errors [error_x, error_y]

    Returns:
        Dict with symmetry metrics:
        - 'mean_x', 'mean_y': Mean errors (should be ~0)
        - 'std_x', 'std_y': Standard deviations
        - 'asymmetry_score': Combined asymmetry metric (0=symmetric, 1=asymmetric)
    """
    mean_x = np.mean(errors[:, 0])
    mean_y = np.mean(errors[:, 1])
    std_x = np.std(errors[:, 0])
    std_y = np.std(errors[:, 1])

    # Asymmetry score: normalized mean error relative to std
    asymmetry_x = abs(mean_x) / (std_x + 1e-6)
    asymmetry_y = abs(mean_y) / (std_y + 1e-6)

    # Combined score (0-1 range via tanh)
    asymmetry_score = np.tanh((asymmetry_x + asymmetry_y) / 2)

    return {
        "mean_x": mean_x,
        "mean_y": mean_y,
        "std_x": std_x,
        "std_y": std_y,
        "asymmetry_score": asymmetry_score,
    }


def refine_principal_point_grid(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    focal: float,
    camera_pose: np.ndarray,
    initial_pp: Tuple[float, float],
    image_size: Tuple[int, int],
    distortion_coeffs: Optional[Dict[str, float]] = None,
    search_radius: float = 0.1,
    grid_steps: int = 11,
) -> PrincipalPointRefinementResult:
    """Refine principal point using 2D grid search.

    This method evaluates principal point candidates on a grid around the
    initial estimate, selecting the one that minimizes reprojection RMSE
    and error asymmetry.

    Args:
        points_3d: Nx3 array of 3D world points
        points_2d: Nx2 array of corresponding 2D image points (normalized coords)
        focal: Focal length (normalized by width)
        camera_pose: 4x4 camera-to-world transformation matrix
        initial_pp: Initial (cx, cy) principal point (normalized coords)
        image_size: (width, height) of image in pixels
        distortion_coeffs: Optional distortion parameters
        search_radius: Search radius in normalized coords (default: 0.1 = 10% of width)
        grid_steps: Number of steps in each direction (default: 11)

    Returns:
        PrincipalPointRefinementResult with refined principal point

    Raises:
        ValueError: If insufficient correspondences (need >= 10 points)
    """
    if len(points_3d) < 10:
        raise ValueError(
            f"Need at least 10 point correspondences for PP refinement, got {len(points_3d)}"
        )

    logger.info(
        f"Grid search for principal point refinement from {len(points_3d)} correspondences"
    )

    # Convert to camera frame
    world_to_cam = np.linalg.inv(camera_pose)
    R = world_to_cam[:3, :3]
    t = world_to_cam[:3, 3]
    points_3d_cam = (R @ points_3d.T).T + t

    # Filter points behind camera
    valid_mask = points_3d_cam[:, 2] > 0
    if valid_mask.sum() < 10:
        raise ValueError(f"Only {valid_mask.sum()} points in front of camera")

    points_3d_cam = points_3d_cam[valid_mask]
    points_2d = points_2d[valid_mask]

    # Compute initial errors
    initial_errors = compute_reprojection_errors(
        points_3d_cam,
        points_2d,
        focal,
        initial_pp,
        image_size,
        distortion_coeffs,
    )
    initial_rmse = np.sqrt(np.mean(initial_errors**2))
    initial_symmetry = analyze_error_symmetry(initial_errors)

    # Grid search
    cx_min = initial_pp[0] - search_radius
    cx_max = initial_pp[0] + search_radius
    cy_min = initial_pp[1] - search_radius
    cy_max = initial_pp[1] + search_radius

    # Clamp to valid range (within ±30% of center is reasonable)
    cx_min = max(cx_min, 0.2)
    cx_max = min(cx_max, 0.8)
    cy_min = max(cy_min, 0.2)
    cy_max = min(cy_max, 0.8)

    cx_range = np.linspace(cx_min, cx_max, grid_steps)
    cy_range = np.linspace(cy_min, cy_max, grid_steps)

    best_pp = initial_pp
    best_rmse = initial_rmse
    best_asymmetry = initial_symmetry["asymmetry_score"]

    for cx in cx_range:
        for cy in cy_range:
            candidate_pp = (cx, cy)

            errors = compute_reprojection_errors(
                points_3d_cam,
                points_2d,
                focal,
                candidate_pp,
                image_size,
                distortion_coeffs,
            )

            rmse = np.sqrt(np.mean(errors**2))
            symmetry = analyze_error_symmetry(errors)

            # Combined score: 70% RMSE reduction + 30% symmetry improvement
            score = 0.7 * rmse + 0.3 * symmetry["asymmetry_score"] * rmse

            if score < (0.7 * best_rmse + 0.3 * best_asymmetry * best_rmse):
                best_pp = candidate_pp
                best_rmse = rmse
                best_asymmetry = symmetry["asymmetry_score"]

    improvement = initial_rmse - best_rmse
    asymmetry_reduction = initial_symmetry["asymmetry_score"] - best_asymmetry

    # Compute confidence
    confidence = np.clip(improvement / (initial_rmse + 1e-6), 0, 1)

    # Bonus for symmetry improvement
    if asymmetry_reduction > 0:
        confidence = min(1.0, confidence * 1.2)

    # Penalize if moved far from center
    distance_from_center = np.sqrt((best_pp[0] - 0.5) ** 2 + (best_pp[1] - 0.5) ** 2)
    if distance_from_center > 0.25:
        confidence *= 0.8
        logger.warning(
            f"Principal point far from center: ({best_pp[0]:.3f}, {best_pp[1]:.3f})"
        )

    logger.info(
        f"Grid search refined PP: ({initial_pp[0]:.3f}, {initial_pp[1]:.3f}) -> "
        f"({best_pp[0]:.3f}, {best_pp[1]:.3f}), RMSE: {initial_rmse:.2f} -> {best_rmse:.2f} px"
    )

    return PrincipalPointRefinementResult(
        original_pp=initial_pp,
        refined_pp=best_pp,
        improvement=improvement,
        iterations=grid_steps * grid_steps,
        converged=True,
        confidence=confidence,
        final_rmse=best_rmse,
        asymmetry_reduction=asymmetry_reduction,
    )


def refine_principal_point_gradient(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    focal: float,
    camera_pose: np.ndarray,
    initial_pp: Tuple[float, float],
    image_size: Tuple[int, int],
    distortion_coeffs: Optional[Dict[str, float]] = None,
) -> PrincipalPointRefinementResult:
    """Refine principal point using gradient-based optimization.

    This method uses Nelder-Mead optimization to minimize reprojection RMSE.
    It's faster than grid search but may get stuck in local minima if the
    initial estimate is poor.

    Args:
        points_3d: Nx3 array of 3D world points
        points_2d: Nx2 array of corresponding 2D image points (normalized coords)
        focal: Focal length (normalized by width)
        camera_pose: 4x4 camera-to-world transformation matrix
        initial_pp: Initial (cx, cy) principal point (normalized coords)
        image_size: (width, height) of image in pixels
        distortion_coeffs: Optional distortion parameters

    Returns:
        PrincipalPointRefinementResult with refined principal point

    Raises:
        ValueError: If insufficient correspondences (need >= 10 points)
    """
    if len(points_3d) < 10:
        raise ValueError(
            f"Need at least 10 point correspondences for PP refinement, got {len(points_3d)}"
        )

    logger.info(
        f"Gradient-based principal point refinement from {len(points_3d)} correspondences"
    )

    # Convert to camera frame
    world_to_cam = np.linalg.inv(camera_pose)
    R = world_to_cam[:3, :3]
    t = world_to_cam[:3, 3]
    points_3d_cam = (R @ points_3d.T).T + t

    # Filter points behind camera
    valid_mask = points_3d_cam[:, 2] > 0
    if valid_mask.sum() < 10:
        raise ValueError(f"Only {valid_mask.sum()} points in front of camera")

    points_3d_cam = points_3d_cam[valid_mask]
    points_2d = points_2d[valid_mask]

    # Compute initial errors
    initial_errors = compute_reprojection_errors(
        points_3d_cam,
        points_2d,
        focal,
        initial_pp,
        image_size,
        distortion_coeffs,
    )
    initial_rmse = np.sqrt(np.mean(initial_errors**2))
    initial_symmetry = analyze_error_symmetry(initial_errors)

    def objective(pp_vec: np.ndarray) -> float:
        """Objective function: RMSE + asymmetry penalty."""
        cx, cy = pp_vec
        candidate_pp = (cx, cy)

        errors = compute_reprojection_errors(
            points_3d_cam,
            points_2d,
            focal,
            candidate_pp,
            image_size,
            distortion_coeffs,
        )

        rmse = np.sqrt(np.mean(errors**2))
        symmetry = analyze_error_symmetry(errors)

        # Combined objective: RMSE + asymmetry penalty
        return rmse + 0.5 * symmetry["asymmetry_score"] * rmse

    # Optimize
    result = minimize(
        objective,
        x0=np.array(initial_pp),
        method="Nelder-Mead",
        options={"maxiter": 100, "xatol": 1e-4},
    )

    refined_pp = tuple(result.x)

    # Clamp to reasonable range
    refined_pp = (
        np.clip(refined_pp[0], 0.2, 0.8),
        np.clip(refined_pp[1], 0.2, 0.8),
    )

    # Compute final errors
    final_errors = compute_reprojection_errors(
        points_3d_cam,
        points_2d,
        focal,
        refined_pp,
        image_size,
        distortion_coeffs,
    )
    final_rmse = np.sqrt(np.mean(final_errors**2))
    final_symmetry = analyze_error_symmetry(final_errors)

    improvement = initial_rmse - final_rmse
    asymmetry_reduction = (
        initial_symmetry["asymmetry_score"] - final_symmetry["asymmetry_score"]
    )

    # Compute confidence
    confidence = np.clip(improvement / (initial_rmse + 1e-6), 0, 1)

    if asymmetry_reduction > 0:
        confidence = min(1.0, confidence * 1.2)

    # Penalize if far from center
    distance_from_center = np.sqrt(
        (refined_pp[0] - 0.5) ** 2 + (refined_pp[1] - 0.5) ** 2
    )
    if distance_from_center > 0.25:
        confidence *= 0.8

    logger.info(
        f"Gradient refined PP: ({initial_pp[0]:.3f}, {initial_pp[1]:.3f}) -> "
        f"({refined_pp[0]:.3f}, {refined_pp[1]:.3f}), RMSE: {initial_rmse:.2f} -> {final_rmse:.2f} px"
    )

    return PrincipalPointRefinementResult(
        original_pp=initial_pp,
        refined_pp=refined_pp,
        improvement=improvement,
        iterations=result.nfev,
        converged=result.success,
        confidence=confidence,
        final_rmse=final_rmse,
        asymmetry_reduction=asymmetry_reduction,
    )


def refine_principal_point_auto(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    focal: float,
    camera_pose: np.ndarray,
    initial_pp: Tuple[float, float],
    image_size: Tuple[int, int],
    distortion_coeffs: Optional[Dict[str, float]] = None,
    method: str = "auto",
) -> PrincipalPointRefinementResult:
    """Automatically select and apply principal point refinement.

    Chooses between grid search and gradient-based optimization:
    - 'grid': Grid search (robust but slower)
    - 'gradient': Gradient-based (fast but needs good initial estimate)
    - 'auto': Grid search if initial PP is suspect, gradient otherwise

    Args:
        points_3d: Nx3 array of 3D world points
        points_2d: Nx2 array of corresponding 2D image points
        focal: Focal length (normalized by width)
        camera_pose: 4x4 camera-to-world transformation matrix
        initial_pp: Initial (cx, cy) principal point (normalized coords)
        image_size: (width, height) in pixels
        distortion_coeffs: Optional distortion parameters
        method: Refinement method ('grid', 'gradient', or 'auto')

    Returns:
        PrincipalPointRefinementResult
    """
    # Check if initial PP is suspect (exactly at center)
    is_default = abs(initial_pp[0] - 0.5) < 0.001 and abs(initial_pp[1] - 0.5) < 0.001

    if method == "auto":
        # Use grid search for suspect defaults, gradient otherwise
        method = "grid" if is_default else "gradient"
        logger.info(f"Auto-selected {method} method (is_default={is_default})")

    if method == "grid":
        return refine_principal_point_grid(
            points_3d,
            points_2d,
            focal,
            camera_pose,
            initial_pp,
            image_size,
            distortion_coeffs,
        )
    else:
        return refine_principal_point_gradient(
            points_3d,
            points_2d,
            focal,
            camera_pose,
            initial_pp,
            image_size,
            distortion_coeffs,
        )
