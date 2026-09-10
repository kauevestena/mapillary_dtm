"""Independent vertical orientation; horizontal-only map registration."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .reconstruction import Reconstruction


def vertical_basis(reference: dict) -> np.ndarray:
    """Rows transform reconstruction vectors into a right-handed level frame.

    `up` points AWAY from gravity. A coordinate-system label, camera pitch,
    EXIF orientation, or a GNSS altitude fit is not a vertical observation.
    """
    if not reference or reference.get("source") not in {
        "calibrated_imu", "survey_control", "validated_verticals"
    } or not str(reference.get("evidence", "")).strip():
        raise ValueError("A documented independent vertical reference is required")
    up = np.asarray(reference.get("up"), dtype=float)
    if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-10:
        raise ValueError("vertical_reference.up must be a finite nonzero 3-vector")
    uncertainty = reference.get("uncertainty_deg")
    if uncertainty is not None and (not np.isfinite(uncertainty) or not 0 <= uncertainty < 45):
        raise ValueError("Vertical uncertainty must be null or an angular bound in [0, 45)")
    up /= np.linalg.norm(up)
    # Deterministic horizontal basis, not an assertion about north.
    axis = np.eye(3)[int(np.argmin(np.abs(up)))]
    x = axis - np.dot(axis, up) * up
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(up, x), up])


def up_from_imu(reconstruction: Reconstruction, samples: list[dict]) -> tuple[np.ndarray, float]:
    """Combine calibrated camera-frame UP measurements, returning angular scatter.

    An accelerometer under vehicle acceleration does not directly measure up.
    Bias/calibration uncertainty must be supplied separately from this scatter.
    """
    if len(samples) < 3:
        raise ValueError("At least three calibrated IMU samples are required")
    vectors = []
    for sample in samples:
        vector = np.asarray(sample["up_camera"], dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-10:
            raise ValueError("Invalid camera-frame IMU up vector")
        vectors.append(reconstruction.shots[sample["image"]].rotation_cw.T @
                       (vector / np.linalg.norm(vector)))
    vectors = np.asarray(vectors)
    center = vectors.mean(axis=0)
    if np.linalg.norm(center) < 0.9:
        raise ValueError("IMU up measurements disagree; check timing and axis conventions")
    center /= np.linalg.norm(center)
    scatter = float(np.max(np.degrees(np.arccos(np.clip(vectors @ center, -1, 1)))))
    return center, scatter


def metric_crs(value):
    from pyproj import CRS

    crs = CRS.from_user_input(value)
    if (not crs.is_projected or len(crs.axis_info) < 2 or
            any(not np.isclose(axis.unit_conversion_factor, 1) for axis in crs.axis_info[:2]) or
            crs.to_epsg() == 3857):
        raise ValueError("Use a local projected CRS in metres (e.g. UTM), not degrees or Web Mercator")
    return crs


@dataclass
class Registration:
    matrix: np.ndarray
    translation: np.ndarray
    scale: float
    rmse_m: float
    rejected_controls: int

    def transform(self, xyz: np.ndarray) -> np.ndarray:
        return np.asarray(xyz, dtype=float) @ self.matrix.T + self.translation


def _similarity(x: np.ndarray, y: np.ndarray):
    xc, yc = x.mean(axis=0), y.mean(axis=0)
    u, v = x - xc, y - yc
    denom = float(np.sum(u * u))
    if denom < 1e-12:
        raise ValueError("Horizontal controls have no baseline")
    a = float(np.sum(u * v) / denom)
    b = float(np.sum(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]) / denom)
    matrix = np.array([[a, -b], [b, a]])
    return matrix, yc - matrix @ xc


def register_horizontal(reconstruction: Reconstruction, basis: np.ndarray,
                        georeference: dict) -> Registration:
    """One isotropic scale + yaw + XY translation. GPS altitude is never read.

    The same scale is applied to Z, preserving all surface angles. Vertical
    translation remains arbitrary. Controls are registered camera centers.
    """
    controls = georeference.get("controls", [])
    names = [row["image"] for row in controls]
    if len(controls) < 3 or len(set(names)) != len(names):
        raise ValueError("Provide at least three distinct horizontal camera controls")
    if not str(georeference.get("source", "")).strip():
        raise ValueError("Horizontal control provenance is required")
    x = np.array([(basis @ reconstruction.shots[row["image"]].center)[:2]
                  for row in controls])
    y = np.array([[row["x"], row["y"]] for row in controls], dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("Horizontal controls contain non-finite coordinates")
    threshold = float(georeference.get("max_residual_m", 3.0))
    min_baseline = float(georeference.get("min_baseline_m", 10.0))
    if not np.isfinite(threshold) or threshold <= 0 or not np.isfinite(min_baseline) or min_baseline <= 0:
        raise ValueError("Control tolerances and baseline must be finite and positive")
    best, best_score = None, (-1, -float("inf"))
    # Evenly sampled pairs bound the work for long sequences, deterministically.
    indices = np.linspace(0, len(x)-1, min(30, len(x))).astype(int)
    for i, j in combinations(indices, 2):
        if np.linalg.norm(y[i] - y[j]) < min_baseline or np.linalg.norm(x[i] - x[j]) < 1e-8:
            continue
        matrix, translation = _similarity(x[[i, j]], y[[i, j]])
        residuals = np.linalg.norm(x @ matrix.T + translation - y, axis=1)
        mask = residuals <= threshold
        score = (int(mask.sum()), -float(np.median(residuals)))
        if score > best_score:
            best, best_score = mask, score
    if best is None or best.sum() < max(3, int(np.ceil(0.7 * len(x)))):
        raise ValueError("Horizontal registration lacks a consistent control baseline")
    matrix, translation = _similarity(x[best], y[best])
    errors = np.linalg.norm(x[best] @ matrix.T + translation - y[best], axis=1)
    scale = float(np.linalg.norm(matrix[0]))
    if not np.isfinite(scale) or scale <= 0 or errors.max() > threshold:
        raise ValueError("Horizontal registration exceeds the residual limit")
    mapping = np.zeros((3, 3))
    mapping[:2, :2], mapping[2, 2] = matrix, scale
    return Registration(mapping @ basis, np.r_[translation, 0.0], scale,
                        float(np.sqrt(np.mean(errors**2))), int((~best).sum()))
