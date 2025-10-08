"""Full self-calibration workflow for camera intrinsic refinement.

This module provides the complete self-calibration pipeline that integrates:
1. Camera parameter validation (Task 1)
2. Focal length refinement (Task 2)
3. Distortion coefficient refinement (Task 3)
4. Principal point refinement (Task 4)

The workflow implements iterative bundle adjustment-style refinement:
- Coordinate descent: Refine one parameter group at a time
- Convergence detection: Stop when improvements fall below threshold
- Quality metrics: Track confidence and RMSE across iterations
- Fallback strategies: Handle failure cases gracefully

Typical usage:
    result = refine_camera_full(
        points_3d, points_2d, camera, camera_pose, image_size
    )
    refined_camera = result.refined_camera
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np

from .camera_refinement import (
    validate_intrinsics,
    validate_sequence_consistency,
    needs_refinement,
)
from .focal_refinement import refine_focal_geometric, refine_focal_ransac
from .distortion_refinement import (
    refine_distortion_auto,
    iterative_refinement as focal_distortion_iterative,
)
from .principal_point_refinement import refine_principal_point_auto

logger = logging.getLogger(__name__)


@dataclass
class SelfCalibrationResult:
    """Complete result from self-calibration workflow."""

    original_camera: Dict
    refined_camera: Dict
    validation_report: Dict
    refinement_history: List[Dict] = field(default_factory=list)
    total_iterations: int = 0
    converged: bool = False
    final_rmse: float = 0.0
    improvement: float = 0.0
    confidence: float = 0.0
    method_used: str = "none"


@dataclass
class IterationResult:
    """Result from a single refinement iteration."""

    iteration: int
    focal: float
    principal_point: Tuple[float, float]
    distortion_coeffs: Dict[str, float]
    rmse: float
    improvement: float
    converged: bool


def compute_rmse(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    focal: float,
    principal_point: Tuple[float, float],
    camera_pose: np.ndarray,
    image_size: Tuple[int, int],
    distortion_coeffs: Optional[Dict[str, float]] = None,
) -> float:
    """Compute reprojection RMSE for given camera parameters.

    Args:
        points_3d: Nx3 array of 3D world points
        points_2d: Nx2 array of 2D image points (normalized coords)
        focal: Focal length (normalized)
        principal_point: (cx, cy) normalized
        camera_pose: 4x4 camera-to-world matrix
        image_size: (width, height) pixels
        distortion_coeffs: Optional distortion parameters

    Returns:
        RMSE in pixels
    """
    from .principal_point_refinement import compute_reprojection_errors

    # Transform to camera frame
    world_to_cam = np.linalg.inv(camera_pose)
    R = world_to_cam[:3, :3]
    t = world_to_cam[:3, 3]
    points_3d_cam = (R @ points_3d.T).T + t

    # Filter points behind camera
    valid_mask = points_3d_cam[:, 2] > 0
    points_3d_cam = points_3d_cam[valid_mask]
    points_2d_valid = points_2d[valid_mask]

    errors = compute_reprojection_errors(
        points_3d_cam,
        points_2d_valid,
        focal,
        principal_point,
        image_size,
        distortion_coeffs,
    )

    return np.sqrt(np.mean(errors**2))


def refine_camera_full(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    camera: Dict,
    camera_pose: np.ndarray,
    image_size: Tuple[int, int],
    max_iterations: int = 5,
    convergence_threshold: float = 0.1,
    use_ransac: bool = False,
    optimize_order: List[str] = None,
) -> SelfCalibrationResult:
    """Complete self-calibration workflow refining all camera parameters.

    This function orchestrates the full refinement pipeline:
    1. Validate initial parameters
    2. Iteratively refine focal, distortion, and principal point
    3. Monitor convergence
    4. Return refined camera with quality metrics

    Args:
        points_3d: Nx3 array of 3D world points
        points_2d: Nx2 array of 2D image points (normalized coords)
        camera: Initial camera parameters dict with keys:
            - 'focal': Focal length (normalized)
            - 'principal_point': [cx, cy] normalized
            - 'projection_type': Camera type string
            - Distortion coefficients (k1, k2, etc.)
        camera_pose: 4x4 camera-to-world transformation matrix
        image_size: (width, height) in pixels
        max_iterations: Maximum refinement iterations (default: 5)
        convergence_threshold: RMSE change threshold in pixels (default: 0.1)
        use_ransac: Whether to use RANSAC for focal (default: False)
        optimize_order: Order of parameters to refine (default: ['focal', 'distortion', 'pp'])

    Returns:
        SelfCalibrationResult with refined parameters and metrics

    Raises:
        ValueError: If insufficient correspondences (need >= 10 points)
    """
    if len(points_3d) < 10:
        raise ValueError(
            f"Need at least 10 correspondences for calibration, got {len(points_3d)}"
        )

    logger.info(f"Starting full self-calibration with {len(points_3d)} correspondences")

    # Validate initial parameters
    validation = validate_intrinsics(camera)
    logger.info(
        f"Initial validation - valid: {validation.valid}, "
        f"confidence: {validation.confidence:.2f}, "
        f"needs_refinement: {validation.needs_refinement}"
    )

    # Extract initial parameters
    current_focal = camera.get("focal", 1.0)
    current_pp = tuple(camera.get("principal_point", [0.5, 0.5]))
    current_distortion = {
        k: v for k, v in camera.items() if k in ["k1", "k2", "k3", "k4", "p1", "p2"]
    }
    projection_type = camera.get("projection_type", "perspective")

    # Compute initial RMSE
    initial_rmse = compute_rmse(
        points_3d,
        points_2d,
        current_focal,
        current_pp,
        camera_pose,
        image_size,
        current_distortion,
    )
    logger.info(f"Initial RMSE: {initial_rmse:.2f} px")

    # Determine optimization order
    if optimize_order is None:
        optimize_order = ["focal", "distortion", "pp"]

    # Iterative refinement
    history = []
    prev_rmse = initial_rmse

    for iteration in range(max_iterations):
        logger.info(f"=== Iteration {iteration + 1}/{max_iterations} ===")

        iteration_improved = False

        for param_type in optimize_order:
            if param_type == "focal":
                # Refine focal length
                try:
                    if use_ransac:
                        focal_result = refine_focal_ransac(
                            points_3d,
                            points_2d,
                            current_focal,
                            camera_pose,
                            current_pp,
                            image_size,
                            current_distortion,
                        )
                    else:
                        focal_result = refine_focal_geometric(
                            points_3d,
                            points_2d,
                            current_focal,
                            camera_pose,
                            current_pp,
                            image_size,
                            current_distortion,
                        )

                    if focal_result.converged and focal_result.improvement > 0:
                        current_focal = focal_result.refined_focal
                        iteration_improved = True
                        logger.info(
                            f"  Focal refined: {focal_result.original_focal:.4f} -> "
                            f"{focal_result.refined_focal:.4f}"
                        )

                except Exception as e:
                    logger.warning(f"  Focal refinement failed: {e}")

            elif param_type == "distortion":
                # Refine distortion coefficients
                try:
                    dist_result = refine_distortion_auto(
                        points_3d,
                        points_2d,
                        current_focal,
                        camera_pose,
                        current_pp,
                        image_size,
                        projection_type,
                        current_distortion,
                        optimize_focal=False,
                    )

                    if dist_result.converged and dist_result.improvement > 0:
                        current_distortion = dist_result.refined_coeffs.copy()
                        iteration_improved = True
                        logger.info(
                            f"  Distortion refined: RMSE improved by {dist_result.improvement:.2f} px"
                        )

                except Exception as e:
                    logger.warning(f"  Distortion refinement failed: {e}")

            elif param_type == "pp":
                # Refine principal point
                try:
                    pp_result = refine_principal_point_auto(
                        points_3d,
                        points_2d,
                        current_focal,
                        camera_pose,
                        current_pp,
                        image_size,
                        current_distortion,
                        method="auto",
                    )

                    if pp_result.converged and pp_result.improvement > 0:
                        current_pp = pp_result.refined_pp
                        iteration_improved = True
                        logger.info(
                            f"  Principal point refined: ({pp_result.original_pp[0]:.3f}, "
                            f"{pp_result.original_pp[1]:.3f}) -> "
                            f"({pp_result.refined_pp[0]:.3f}, {pp_result.refined_pp[1]:.3f})"
                        )

                except Exception as e:
                    logger.warning(f"  Principal point refinement failed: {e}")

        # Compute RMSE after iteration
        current_rmse = compute_rmse(
            points_3d,
            points_2d,
            current_focal,
            current_pp,
            camera_pose,
            image_size,
            current_distortion,
        )

        improvement = prev_rmse - current_rmse

        # Record iteration
        iter_result = IterationResult(
            iteration=iteration + 1,
            focal=current_focal,
            principal_point=current_pp,
            distortion_coeffs=current_distortion.copy(),
            rmse=current_rmse,
            improvement=improvement,
            converged=abs(improvement) < convergence_threshold,
        )
        history.append(iter_result)

        logger.info(
            f"  Iteration RMSE: {current_rmse:.2f} px (change: {improvement:+.2f} px)"
        )

        # Check convergence
        if abs(improvement) < convergence_threshold:
            logger.info(f"Converged after {iteration + 1} iterations")
            break

        if not iteration_improved:
            logger.info(
                f"No improvement possible, stopping at iteration {iteration + 1}"
            )
            break

        prev_rmse = current_rmse

    # Build refined camera
    refined_camera = camera.copy()
    refined_camera["focal"] = current_focal
    refined_camera["principal_point"] = list(current_pp)
    refined_camera.update(current_distortion)

    # Compute final metrics
    total_improvement = initial_rmse - current_rmse
    confidence = np.clip(total_improvement / (initial_rmse + 1e-6), 0, 1)

    # Bonus for low final RMSE
    if current_rmse < 1.0:
        confidence = min(1.0, confidence * 1.2)

    converged = len(history) > 0 and history[-1].converged

    logger.info(
        f"Self-calibration complete: RMSE {initial_rmse:.2f} -> {current_rmse:.2f} px, "
        f"improvement: {total_improvement:.2f} px, confidence: {confidence:.2f}"
    )

    return SelfCalibrationResult(
        original_camera=camera,
        refined_camera=refined_camera,
        validation_report={
            "initial": validation.__dict__,
            "final_rmse": current_rmse,
            "initial_rmse": initial_rmse,
        },
        refinement_history=[h.__dict__ for h in history],
        total_iterations=len(history),
        converged=converged,
        final_rmse=current_rmse,
        improvement=total_improvement,
        confidence=confidence,
        method_used="full_iterative",
    )


def refine_camera_quick(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    camera: Dict,
    camera_pose: np.ndarray,
    image_size: Tuple[int, int],
) -> SelfCalibrationResult:
    """Quick single-pass refinement for computationally constrained scenarios.

    This simplified workflow:
    1. Validates parameters
    2. Refines focal only (fastest improvement)
    3. Optionally refines principal point if default detected
    4. Skips distortion (most expensive)

    Use when:
    - Computational budget is limited
    - Initial parameters are reasonably good
    - Quick improvement needed for online processing

    Args:
        points_3d: Nx3 array of 3D world points
        points_2d: Nx2 array of 2D image points
        camera: Initial camera parameters
        camera_pose: 4x4 camera-to-world matrix
        image_size: (width, height) pixels

    Returns:
        SelfCalibrationResult with focal (and possibly PP) refined
    """
    logger.info("Quick self-calibration (focal + PP only)")

    validation = validate_intrinsics(camera)

    focal = camera.get("focal", 1.0)
    pp = tuple(camera.get("principal_point", [0.5, 0.5]))
    distortion = {k: v for k, v in camera.items() if k in ["k1", "k2", "p1", "p2"]}

    initial_rmse = compute_rmse(
        points_3d, points_2d, focal, pp, camera_pose, image_size, distortion
    )

    # Refine focal
    try:
        focal_result = refine_focal_geometric(
            points_3d, points_2d, focal, camera_pose, pp, image_size, distortion
        )
        if focal_result.converged:
            focal = focal_result.refined_focal
    except Exception as e:
        logger.warning(f"Focal refinement failed: {e}")

    # Refine PP if default detected
    is_default = abs(pp[0] - 0.5) < 0.001 and abs(pp[1] - 0.5) < 0.001
    if is_default:
        try:
            pp_result = refine_principal_point_auto(
                points_3d,
                points_2d,
                focal,
                camera_pose,
                pp,
                image_size,
                distortion,
                method="gradient",
            )
            if pp_result.converged:
                pp = pp_result.refined_pp
        except Exception as e:
            logger.warning(f"PP refinement failed: {e}")

    final_rmse = compute_rmse(
        points_3d, points_2d, focal, pp, camera_pose, image_size, distortion
    )

    refined_camera = camera.copy()
    refined_camera["focal"] = focal
    refined_camera["principal_point"] = list(pp)

    improvement = initial_rmse - final_rmse
    confidence = np.clip(improvement / (initial_rmse + 1e-6), 0, 1)

    logger.info(f"Quick calibration: RMSE {initial_rmse:.2f} -> {final_rmse:.2f} px")

    return SelfCalibrationResult(
        original_camera=camera,
        refined_camera=refined_camera,
        validation_report={"initial": validation.__dict__},
        refinement_history=[],
        total_iterations=1,
        converged=True,
        final_rmse=final_rmse,
        improvement=improvement,
        confidence=confidence,
        method_used="quick",
    )


def refine_sequence_cameras(
    sequence_data: Dict[str, Dict],
    correspondences: Dict[str, Tuple[np.ndarray, np.ndarray]],
    poses: Dict[str, np.ndarray],
    image_sizes: Dict[str, Tuple[int, int]],
    method: str = "full",
) -> Dict[str, SelfCalibrationResult]:
    """Refine camera parameters for multiple images in a sequence.

    This function processes a sequence of images, refining each camera
    independently while maintaining sequence-level consistency checks.

    Args:
        sequence_data: Dict mapping image_id -> camera parameters
        correspondences: Dict mapping image_id -> (points_3d, points_2d)
        poses: Dict mapping image_id -> camera pose (4x4 matrix)
        image_sizes: Dict mapping image_id -> (width, height)
        method: Refinement method ('full', 'quick')

    Returns:
        Dict mapping image_id -> SelfCalibrationResult
    """
    logger.info(f"Refining {len(sequence_data)} cameras in sequence")

    results = {}

    for image_id, camera in sequence_data.items():
        if image_id not in correspondences or image_id not in poses:
            logger.warning(f"Missing data for {image_id}, skipping")
            continue

        points_3d, points_2d = correspondences[image_id]
        pose = poses[image_id]
        image_size = image_sizes.get(image_id, (1920, 1080))

        if len(points_3d) < 10:
            logger.warning(
                f"{image_id}: insufficient correspondences ({len(points_3d)})"
            )
            continue

        try:
            if method == "quick":
                result = refine_camera_quick(
                    points_3d, points_2d, camera, pose, image_size
                )
            else:
                result = refine_camera_full(
                    points_3d, points_2d, camera, pose, image_size
                )

            results[image_id] = result

        except Exception as e:
            logger.error(f"{image_id}: Refinement failed: {e}")

    # Sequence-level consistency check
    if results:
        focals = [r.refined_camera["focal"] for r in results.values()]
        consistency = validate_sequence_consistency(
            {k: v.refined_camera for k, v in results.items()}
        )

        logger.info(
            f"Sequence refinement complete: {len(results)} cameras, "
            f"focal range: [{min(focals):.3f}, {max(focals):.3f}], "
            f"consistent: {consistency['consistent']}"
        )

    return results
