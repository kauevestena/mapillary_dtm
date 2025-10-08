"""
Camera intrinsic parameter refinement (self-calibration).

This module provides tools to validate and refine camera intrinsic parameters
(focal length, principal point, distortion coefficients) for improved
reconstruction accuracy, especially for fisheye and spherical cameras.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from .. import constants
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    import constants

log = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Camera intrinsic parameter validation results."""

    valid: bool
    warnings: List[str]
    needs_refinement: bool
    confidence: float  # 0.0 - 1.0
    focal_status: str  # 'good', 'suspect', 'missing'
    distortion_status: str
    principal_point_status: str


def validate_intrinsics(
    camera: Dict,
    image_id: Optional[str] = None,
    expected_focal_range: Tuple[float, float] = (0.5, 1.5),
) -> ValidationReport:
    """Validate camera intrinsic parameters and flag suspicious values.

    Checks focal length, principal point, and distortion coefficients against
    reasonable ranges and common manufacturer defaults.

    Args:
        camera: Camera dictionary (OpenSfM format)
        image_id: Optional image identifier for logging
        expected_focal_range: (min, max) normalized focal length bounds

    Returns:
        ValidationReport with validation status and warnings
    """
    warnings = []
    needs_refinement = False
    confidence = 1.0

    # Extract parameters
    projection_type = camera.get("projection_type", "perspective")
    focal = camera.get("focal")
    focal_y = camera.get("focal_y")
    pp = camera.get("principal_point", [0.0, 0.0])
    width = camera.get("width", 1920)
    height = camera.get("height", 1080)

    # Validate focal length
    focal_status = "good"
    if focal is None:
        warnings.append("Missing focal length - will use default")
        focal_status = "missing"
        needs_refinement = True
        confidence *= 0.5
    else:
        # Check if focal is reasonable
        if focal < expected_focal_range[0]:
            warnings.append(
                f"Focal length {focal:.3f} unusually low (< {expected_focal_range[0]})"
            )
            focal_status = "suspect"
            needs_refinement = True
            confidence *= 0.7
        elif focal > expected_focal_range[1]:
            warnings.append(
                f"Focal length {focal:.3f} unusually high (> {expected_focal_range[1]})"
            )
            focal_status = "suspect"
            needs_refinement = True
            confidence *= 0.7

        # Check for common manufacturer default (exactly 1.0)
        if abs(focal - 1.0) < 1e-6:
            warnings.append("Focal length is exactly 1.0 - likely manufacturer default")
            focal_status = "suspect"
            needs_refinement = True
            confidence *= 0.8

        # Check for suspicious round numbers
        if abs(focal - round(focal, 1)) < 1e-6 and focal != 0.5:
            warnings.append(
                f"Focal length {focal} is suspiciously round - may be approximate"
            )
            confidence *= 0.9

    # Validate focal_y (if present)
    if focal_y is not None:
        if focal is not None:
            aspect_ratio = focal_y / focal
            if abs(aspect_ratio - 1.0) > 0.05:  # More than 5% difference
                warnings.append(
                    f"Focal aspect ratio {aspect_ratio:.3f} deviates from 1.0"
                )
                confidence *= 0.95

    # Validate principal point
    pp_status = "good"
    if pp:
        cx, cy = pp[0], pp[1]

        # Principal point should be near image center (within ±20%)
        if abs(cx - 0.5) > 0.2:
            warnings.append(f"Principal point x={cx:.3f} far from center (0.5)")
            pp_status = "suspect"
            needs_refinement = True
            confidence *= 0.85

        if abs(cy - 0.5) > 0.2:
            warnings.append(f"Principal point y={cy:.3f} far from center (0.5)")
            pp_status = "suspect"
            needs_refinement = True
            confidence *= 0.85

        # Check for exact center (may be default)
        if abs(cx - 0.5) < 1e-6 and abs(cy - 0.5) < 1e-6:
            warnings.append("Principal point is exactly at center - may be default")
            pp_status = "suspect"
            confidence *= 0.9
    else:
        warnings.append("Principal point not specified - will use image center")
        pp_status = "missing"
        confidence *= 0.95

    # Validate distortion coefficients
    distortion_status = "good"
    distortion_params = ["k1", "k2", "k3", "p1", "p2", "k4", "k5", "k6"]
    distortion_present = {
        k: camera.get(k) for k in distortion_params if camera.get(k) is not None
    }

    if not distortion_present:
        if projection_type in ["fisheye", "brown"]:
            warnings.append(f"No distortion coefficients for {projection_type} camera")
            distortion_status = "missing"
            needs_refinement = True
            confidence *= 0.7
        # Spherical cameras don't need distortion
    else:
        # Check coefficient magnitudes
        for coef_name, coef_val in distortion_present.items():
            max_expected = {
                "k1": 1.0,
                "k2": 0.5,
                "k3": 0.2,
                "p1": 0.1,
                "p2": 0.1,
                "k4": 0.1,
                "k5": 0.1,
                "k6": 0.05,
            }

            threshold = max_expected.get(coef_name, 1.0)
            if abs(coef_val) > threshold:
                warnings.append(
                    f"Distortion {coef_name}={coef_val:.4f} exceeds typical range (|{coef_name}| < {threshold})"
                )
                distortion_status = "suspect"
                needs_refinement = True
                confidence *= 0.8

        # Check for all zeros (likely uninitialized)
        if all(abs(v) < 1e-9 for v in distortion_present.values()):
            warnings.append(
                "All distortion coefficients are zero - may be uninitialized"
            )
            distortion_status = "suspect"
            needs_refinement = True
            confidence *= 0.7

    # Overall validation
    valid = (
        focal_status in ["good", "missing"]
        and pp_status in ["good", "missing"]
        and distortion_status in ["good", "missing"]
    )

    if not valid:
        warnings.append(
            "Camera parameters have suspicious values - refinement recommended"
        )

    log.debug(
        f"Camera validation{' for ' + image_id if image_id else ''}: "
        f"valid={valid}, confidence={confidence:.2f}, warnings={len(warnings)}"
    )

    return ValidationReport(
        valid=valid,
        warnings=warnings,
        needs_refinement=needs_refinement,
        confidence=confidence,
        focal_status=focal_status,
        distortion_status=distortion_status,
        principal_point_status=pp_status,
    )


def validate_sequence_consistency(
    cameras: Dict[str, Dict], tolerance: float = 0.1
) -> Dict[str, any]:
    """Check consistency of camera parameters within a sequence.

    Detects sudden parameter changes or outliers that may indicate
    bad metadata or mixed camera types.

    Args:
        cameras: Dict of image_id -> camera dict
        tolerance: Maximum allowed parameter variation (fraction)

    Returns:
        Dict with consistency statistics and warnings
    """
    if not cameras:
        return {"consistent": True, "warnings": [], "stats": {}}

    warnings = []

    # Extract focal lengths
    focals = []
    for img_id, cam in cameras.items():
        focal = cam.get("focal")
        if focal is not None:
            focals.append((img_id, focal))

    if not focals:
        warnings.append("No focal lengths found in sequence")
        return {"consistent": False, "warnings": warnings, "stats": {}}

    # Compute statistics
    focal_values = [f for _, f in focals]
    focal_mean = np.mean(focal_values)
    focal_std = np.std(focal_values)
    focal_min = np.min(focal_values)
    focal_max = np.max(focal_values)
    focal_range = focal_max - focal_min

    stats = {
        "focal_mean": float(focal_mean),
        "focal_std": float(focal_std),
        "focal_min": float(focal_min),
        "focal_max": float(focal_max),
        "focal_range": float(focal_range),
        "n_cameras": len(focals),
    }

    # Check consistency
    consistent = True

    # Check if focal lengths vary too much
    if focal_mean > 0:
        relative_range = focal_range / focal_mean
        if relative_range > tolerance:
            warnings.append(
                f"Focal length varies by {relative_range*100:.1f}% across sequence "
                f"(tolerance: {tolerance*100:.1f}%)"
            )
            consistent = False

    # Detect outliers (> 2 sigma from mean)
    if focal_std > 0:
        outliers = []
        for img_id, focal in focals:
            z_score = abs(focal - focal_mean) / focal_std
            if z_score > 2.0:
                outliers.append((img_id, focal, z_score))

        if outliers:
            warnings.append(f"Found {len(outliers)} focal length outliers (> 2σ)")
            consistent = False
            stats["outliers"] = [
                (img_id, float(f), float(z)) for img_id, f, z in outliers[:5]
            ]

    # Check for suspicious jumps between consecutive frames
    if len(focals) > 1:
        max_jump = 0.0
        jump_location = None
        for i in range(len(focals) - 1):
            img_id1, f1 = focals[i]
            img_id2, f2 = focals[i + 1]
            jump = abs(f2 - f1) / max(f1, 1e-9)
            if jump > max_jump:
                max_jump = jump
                jump_location = (img_id1, img_id2)

        if max_jump > tolerance / 2:  # Half tolerance for consecutive frames
            warnings.append(
                f"Large focal length jump ({max_jump*100:.1f}%) between frames "
                f"{jump_location[0]} and {jump_location[1]}"
            )
            consistent = False
            stats["max_jump"] = float(max_jump)

    return {"consistent": consistent, "warnings": warnings, "stats": stats}


def needs_refinement(validation: ValidationReport) -> bool:
    """Determine if camera parameters should be refined based on validation.

    Args:
        validation: ValidationReport from validate_intrinsics()

    Returns:
        True if refinement is recommended
    """
    # Refinement recommended if:
    # 1. Explicit flag set
    if validation.needs_refinement:
        return True

    # 2. Low confidence score
    if validation.confidence < 0.8:
        return True

    # 3. Multiple warnings
    if len(validation.warnings) >= 3:
        return True

    # 4. Any 'suspect' or 'missing' status
    if validation.focal_status in ["suspect", "missing"]:
        return True
    if validation.distortion_status in ["suspect", "missing"]:
        return True

    return False


def summarize_validation(
    validations: Dict[str, ValidationReport], sequence_consistency: Dict[str, any]
) -> Dict[str, any]:
    """Create summary statistics from multiple camera validations.

    Args:
        validations: Dict of image_id -> ValidationReport
        sequence_consistency: Output from validate_sequence_consistency()

    Returns:
        Summary dict with aggregate statistics
    """
    if not validations:
        return {"total_cameras": 0, "needs_refinement": False}

    total = len(validations)
    valid_count = sum(1 for v in validations.values() if v.valid)
    needs_refine_count = sum(1 for v in validations.values() if v.needs_refinement)

    # Aggregate confidence
    confidences = [v.confidence for v in validations.values()]
    avg_confidence = np.mean(confidences)
    min_confidence = np.min(confidences)

    # Aggregate warnings
    all_warnings = []
    for v in validations.values():
        all_warnings.extend(v.warnings)
    unique_warnings = list(set(all_warnings))

    # Status counts
    focal_statuses = [v.focal_status for v in validations.values()]
    distortion_statuses = [v.distortion_status for v in validations.values()]

    summary = {
        "total_cameras": total,
        "valid_count": valid_count,
        "needs_refinement_count": needs_refine_count,
        "needs_refinement": needs_refine_count > total * 0.5,  # More than 50%
        "avg_confidence": float(avg_confidence),
        "min_confidence": float(min_confidence),
        "total_warnings": len(all_warnings),
        "unique_warnings": unique_warnings,
        "focal_good": focal_statuses.count("good"),
        "focal_suspect": focal_statuses.count("suspect"),
        "focal_missing": focal_statuses.count("missing"),
        "distortion_good": distortion_statuses.count("good"),
        "distortion_suspect": distortion_statuses.count("suspect"),
        "distortion_missing": distortion_statuses.count("missing"),
        "sequence_consistent": sequence_consistency.get("consistent", True),
    }

    log.info(
        f"Validation summary: {needs_refine_count}/{total} cameras need refinement, "
        f"avg confidence: {avg_confidence:.2f}, "
        f"sequence consistent: {summary['sequence_consistent']}"
    )

    return summary
