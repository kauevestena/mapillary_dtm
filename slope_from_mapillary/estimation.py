"""Local surface gradients with measured support and explicit rejection gates."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree


@dataclass(frozen=True)
class FitPolicy:
    radius_m: float = 2.0
    min_points: int = 12
    min_span_m: float = 0.5
    max_gap_m: float = 0.75
    max_residual_m: float = 0.04
    min_inlier_fraction: float = 0.85
    max_condition: float = 100.0
    max_slope_deg: float = 40.0
    max_fit_precision_deg: float = 2.0

    def __post_init__(self):
        if not isinstance(self.min_points, int) or self.min_points < 6:
            raise ValueError("min_points must be an integer >= 6")
        for name in ("radius_m", "min_span_m", "max_gap_m", "max_residual_m",
                     "max_condition", "max_fit_precision_deg"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.5 < self.min_inlier_fraction <= 1 or not 0 < self.max_slope_deg < 90:
            raise ValueError("Invalid inlier fraction or slope limit")


@dataclass
class Plane:
    gradient: np.ndarray  # dz/dE, dz/dN; dimensionless
    normal: np.ndarray  # upward-facing unit normal in E,N,U
    residual_rmse_m: float
    fit_precision_deg: float  # internal precision, not field accuracy
    point_count: int
    inlier_fraction: float
    hull: np.ndarray  # halfspace equations in centered horizontal coordinates
    origin: np.ndarray

    @property
    def slope_pct(self):
        return float(100 * np.linalg.norm(self.gradient))

    @property
    def slope_deg(self):
        return float(np.degrees(np.arctan(np.linalg.norm(self.gradient))))


def fit_plane(points: np.ndarray, center: np.ndarray, policy: FitPolicy,
              *, cell_size_m: float = 0) -> tuple[Plane | None, str]:
    """Fit a local plane; never infer a normal from a line-like point cluster.

    Coordinates must share isotropic metre units in a vertically referenced
    frame. Centering XY/Z avoids sensitivity to arbitrary absolute elevation.
    IRLS rejects a small outlier fraction; discontinuities/curbs are rejected
    instead of joined by a smoothed elevation raster.
    """
    points, center = np.asarray(points, dtype=float), np.asarray(center, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or center.shape != (2,):
        raise ValueError("Expected points (N,3) and horizontal center (2,)")
    if not np.isfinite(points).all() or not np.isfinite(center).all():
        raise ValueError("Non-finite plane input")
    if not np.isfinite(cell_size_m) or cell_size_m < 0:
        raise ValueError("cell_size_m must be finite and nonnegative")
    if len(points) < policy.min_points:
        return None, "insufficient_points"
    xy, z = points[:, :2] - center, points[:, 2] - np.median(points[:, 2])
    if np.linalg.norm(xy, axis=1).min() > policy.max_gap_m:
        return None, "unsupported_center"
    design = np.column_stack([xy, np.ones(len(xy))])
    coeff = np.linalg.lstsq(design, z, rcond=None)[0]
    for _ in range(12):
        residual = z - design @ coeff
        robust_scale = max(float(1.4826 * np.median(np.abs(residual - np.median(residual)))), 1e-9)
        weights = np.minimum(1.0, 1.345 * robust_scale / np.maximum(np.abs(residual), 1e-12))
        root = np.sqrt(weights)
        updated = np.linalg.lstsq(design * root[:, None], z * root, rcond=None)[0]
        if np.linalg.norm(updated - coeff) < 1e-10:
            coeff = updated
            break
        coeff = updated
    inliers = np.abs(z - design @ coeff) <= policy.max_residual_m
    fraction = float(inliers.mean())
    if fraction < policy.min_inlier_fraction or inliers.sum() < policy.min_points:
        return None, "nonplanar_or_discontinuous"
    x, values = design[inliers], z[inliers]
    centered = xy[inliers] - xy[inliers].mean(axis=0)
    eigenvalues = np.linalg.eigvalsh(centered.T @ centered / len(centered))
    if eigenvalues[0] < (policy.min_span_m / 4)**2:
        return None, "insufficient_2d_support"
    if eigenvalues[1] / eigenvalues[0] > policy.max_condition:
        return None, "ill_conditioned_support"
    coeff = np.linalg.lstsq(x, values, rcond=None)[0]
    residual = values - x @ coeff
    # Check the entire cell against the observed inlier hull. No extrapolation.
    try:
        hull = ConvexHull(xy[inliers]).equations
    except QhullError:
        return None, "insufficient_2d_support"
    corners = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]) * cell_size_m / 2
    if np.any(corners @ hull[:, :2].T + hull[:, 2] > 1e-9):
        return None, "outside_observed_support"
    # Reject cells straddling gaps even when their convex hull spans the gap.
    if cKDTree(xy[inliers]).query(np.vstack([[0, 0], corners]))[0].max() > policy.max_gap_m:
        return None, "unsupported_cell"
    gradient = coeff[:2]
    slope = float(np.degrees(np.arctan(np.linalg.norm(gradient))))
    if slope > policy.max_slope_deg:
        return None, "steep_or_vertical_surface"
    variance = float(np.sum(residual**2) / (len(values) - 3))
    covariance = variance * np.linalg.inv(x.T @ x)[:2, :2]
    # Largest one-sigma gradient direction, converted to an angle. This omits
    # SfM correlations, calibration bias and vertical-reference error.
    precision = float(np.degrees(np.arctan(np.sqrt(max(np.linalg.eigvalsh(covariance))))))
    if precision > policy.max_fit_precision_deg:
        return None, "imprecise_fit"
    normal = np.r_[-gradient, 1.0]
    normal /= np.linalg.norm(normal)
    return Plane(gradient, normal, float(np.sqrt(np.mean(residual**2))), precision,
                 int(inliers.sum()), fraction, hull, center), "accepted"


def directional_grades(gradient: np.ndarray, tangent: np.ndarray) -> tuple[float, float]:
    """Percent rise along travel and toward its RIGHT side.

    Reversing travel negates both signed grades; their magnitudes stay equal.
    """
    gradient, tangent = np.asarray(gradient, dtype=float), np.asarray(tangent, dtype=float)
    if gradient.shape != (2,) or tangent.shape != (2,) or not np.isfinite([gradient, tangent]).all():
        raise ValueError("Gradient and tangent must be finite horizontal 2-vectors")
    length = np.linalg.norm(tangent)
    if length < 1e-12:
        raise ValueError("Travel direction must have nonzero horizontal length")
    tangent = tangent / length
    right = np.array([tangent[1], -tangent[0]])
    return float(100 * gradient @ tangent), float(100 * gradient @ right)


def angular_bound(plane: Plane, vertical_deg: float | None,
                  geometry_deg: float | None) -> float | None:
    """Conservative screening allowance, not a calibrated confidence interval."""
    if vertical_deg is None or geometry_deg is None:
        return None
    if not np.isfinite([vertical_deg, geometry_deg]).all() or min(vertical_deg, geometry_deg) < 0:
        raise ValueError("Angular uncertainty terms must be finite and nonnegative")
    return float(vertical_deg + geometry_deg + 1.96 * plane.fit_precision_deg)


def screening_status(value_pct: float, bound_deg: float | None, threshold_pct: float) -> str:
    """User-selected screening thresholds are not accessibility certification."""
    if not np.isfinite(value_pct) or not np.isfinite(threshold_pct) or threshold_pct < 0:
        raise ValueError("Grade/threshold must be finite, with a nonnegative threshold")
    if bound_deg is None:
        return "uncertainty_unknown"
    if not np.isfinite(bound_deg) or bound_deg < 0:
        raise ValueError("Angular bound must be finite and nonnegative")
    angle = float(np.degrees(np.arctan(abs(value_pct) / 100)))
    lower = 100 * np.tan(np.radians(max(0, angle - bound_deg)))
    upper = (float("inf") if angle + bound_deg >= 90 else
             100 * np.tan(np.radians(angle + bound_deg)))
    if lower > threshold_pct:
        return "above_threshold"
    if upper <= threshold_pct:
        return "below_threshold"
    return "uncertain"


def directional_allowance(gradient: np.ndarray, tangent: np.ndarray,
                          normal_allowance_deg: float | None) -> float | None:
    """Project a normal's angular cone into a directional slope plane.

    A normal-angle bound alone can understate component uncertainty when a
    surface also tilts in the perpendicular direction.
    """
    if normal_allowance_deg is None:
        return None
    if not np.isfinite(normal_allowance_deg) or normal_allowance_deg < 0:
        raise ValueError("Angular allowance must be finite and nonnegative")
    gradient, tangent = np.asarray(gradient, dtype=float), np.asarray(tangent, dtype=float)
    directional_grades(gradient, tangent)  # validate shapes and nonzero direction
    tangent = tangent / np.linalg.norm(tangent)
    normal = np.r_[-gradient, 1.0]
    normal /= np.linalg.norm(normal)
    projected_length = float(np.hypot(normal[:2] @ tangent, normal[2]))
    if normal_allowance_deg >= 90 or np.sin(np.radians(normal_allowance_deg)) >= projected_length:
        return 90.0
    return float(np.degrees(np.arcsin(np.sin(np.radians(normal_allowance_deg)) / projected_length)))
