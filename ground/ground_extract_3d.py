"""
Label and filter 3D points to ground-only via multi-view semantic voting.

This synthetic implementation blends three sources:

1. Sparse SfM points delivered by the geometry stack.
2. Monocular depth priors (converted into local ground samples).
3. Plane-sweep densification along the vehicle corridor.

Each candidate point is validated against nearby camera poses and cached
ground masks, producing :class:`GroundPoint` records enriched with QA fields
such as semantic probability, triangulation angle and uncertainty.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .. import constants
from ..common_core import FrameMeta, GroundPoint, Pose, ReconstructionResult
from ..depth.monodepth import predict_depths
from ..depth.plane_sweep_ground import SweepResult, sweep

ASSUMED_CAM_HEIGHT = 1.6  # meters
SUPPORT_RADIUS_M = 12.0
LANE_HALF_WIDTH_M = 3.5


def label_and_filter_points(
    recon: Mapping[str, ReconstructionResult],
    scales: Mapping[str, float],
    mask_dir: Path | str = Path("cache/masks"),
    mono_cache: Path | str = Path("cache/depth_mono"),
    *,
    include_sparse: bool = True,
    include_monodepth: bool = True,
    include_plane_sweep: bool = True,
    vo_recon: Mapping[str, ReconstructionResult] | None = None,
) -> List[GroundPoint]:
    """Return ground-only point samples enriched with QA metadata."""

    if not recon:
        return []

    frame_index: Dict[str, Sequence[FrameMeta]] = {
        seq_id: result.frames for seq_id, result in recon.items() if result.frames
    }
    if include_monodepth:
        mono_depths = predict_depths(frame_index, out_dir=mono_cache)
    else:
        mono_depths = {}

    mask_dir_path = Path(mask_dir)
    collected: List[GroundPoint] = []

    for seq_id, result in recon.items():
        frames = list(result.frames or [])
        if not frames:
            continue

        pose_map = result.poses or {}
        if not pose_map:
            continue

        scale = float(scales.get(seq_id, 1.0))
        centers, headings = _camera_centers_and_headings(frames, pose_map, scale)
        if not centers:
            continue

        mask_priors = {
            frame.image_id: _mask_prior(_load_mask(mask_dir_path, frame.image_id))
            for frame in frames
            if frame.image_id in centers
        }
        ordered_frames = [frame for frame in frames if frame.image_id in centers]
        mono_data = mono_depths.get(seq_id, {}) if include_monodepth else {}
        vo_result = vo_recon.get(seq_id) if vo_recon else None

        candidates: List[Tuple[np.ndarray, float, str]] = []

        base_xyz = result.points_xyz if result.points_xyz is not None else np.zeros((0, 3), dtype=float)
        base_points = np.asarray(base_xyz, dtype=np.float64)
        if include_sparse and base_points.size:
            for pt in base_points * scale:
                candidates.append((pt, 0.18, result.source or "recon"))

        if include_monodepth:
            candidates.extend(
                _monodepth_candidates(
                    ordered_frames,
                    centers,
                    headings,
                    mono_data,
                )
            )
        if include_plane_sweep:
            candidates.extend(
                _plane_sweep_candidates(
                    ordered_frames,
                    centers,
                    scale,
                )
            )
        if vo_result is not None:
            candidates.extend(_vo_candidates(vo_result, scale))

        seq_points = _evaluate_candidates(
            seq_id,
            ordered_frames,
            centers,
            mask_priors,
            candidates,
        )
        collected.extend(seq_points)

    return collected


def _camera_centers_and_headings(
    frames: Sequence[FrameMeta],
    pose_map: Mapping[str, Pose],
    scale: float,
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    centers: Dict[str, np.ndarray] = {}
    for frame in frames:
        pose = pose_map.get(frame.image_id)
        if pose is None:
            continue
        centers[frame.image_id] = np.asarray(pose.t, dtype=np.float64) * scale

    headings: Dict[str, np.ndarray] = {}
    ordered = [frame.image_id for frame in frames if frame.image_id in centers]
    for idx, image_id in enumerate(ordered):
        if len(ordered) == 1:
            headings[image_id] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            continue
        if idx < len(ordered) - 1:
            vec = centers[ordered[idx + 1]] - centers[image_id]
        else:
            vec = centers[image_id] - centers[ordered[idx - 1]]
        headings[image_id] = _normalize(vec)
    return centers, headings


def _monodepth_candidates(
    frames: Sequence[FrameMeta],
    centers: Mapping[str, np.ndarray],
    headings: Mapping[str, np.ndarray],
    mono_data: Mapping[str, Mapping[str, np.ndarray]],
) -> List[Tuple[np.ndarray, float, str]]:
    """Convert mono-depth grids to ground point hypotheses."""

    candidates: List[Tuple[np.ndarray, float, str]] = []
    if not mono_data:
        return candidates

    for frame in frames:
        cache = mono_data.get(frame.image_id)
        center = centers.get(frame.image_id)
        heading = headings.get(frame.image_id)
        if not cache or center is None or heading is None:
            continue

        depth = np.asarray(cache.get("depth"), dtype=np.float32)
        uncert = np.asarray(cache.get("uncertainty"), dtype=np.float32)
        if depth.ndim != 2 or depth.size == 0:
            continue

        left = _normalize(np.array([-heading[1], heading[0], 0.0], dtype=np.float64))
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        lane_span = LANE_HALF_WIDTH_M * 2.0

        step_r = max(1, depth.shape[0] // 20)
        step_c = max(1, depth.shape[1] // 24)

        for row in range(0, depth.shape[0], step_r):
            for col in range(0, depth.shape[1], step_c):
                d = float(depth[row, col])
                if not np.isfinite(d) or d <= 0.5:
                    continue
                lat_frac = (col / max(depth.shape[1] - 1, 1)) - 0.5
                lateral = left * (lat_frac * lane_span)
                forward = heading * d * 0.8  # down-weight to avoid overshooting
                point = center + forward + lateral - up * ASSUMED_CAM_HEIGHT
                base_uncert = float(uncert[row, col]) if uncert.shape == depth.shape else float(np.mean(uncert))
                candidates.append((point.astype(np.float64), max(0.12, min(base_uncert, 0.45)), "mono"))
    return candidates


def _plane_sweep_candidates(
    frames: Sequence[FrameMeta],
    centers: Mapping[str, np.ndarray],
    scale: float,
) -> List[Tuple[np.ndarray, float, str]]:
    if len(frames) < 2:
        return []

    candidates: List[Tuple[np.ndarray, float, str]] = []
    for idx in range(len(frames) - 1):
        res: SweepResult = sweep((frames[idx], frames[idx + 1]))
        pts = np.asarray(res.points, dtype=np.float64)
        if pts.size == 0:
            continue
        weights = np.asarray(res.weights, dtype=np.float32)
        pts = pts * float(scale)
        for point, weight in zip(pts, weights, strict=False):
            base_uncert = 0.28 - 0.18 * float(np.clip(weight, 0.0, 1.0))
            candidates.append((point.astype(np.float64), max(0.1, base_uncert), "plane_sweep"))
    return candidates


def _vo_candidates(
    vo_result: ReconstructionResult,
    scale: float,
) -> List[Tuple[np.ndarray, float, str]]:
    frames = list(vo_result.frames or [])
    if len(frames) < 2:
        return []

    positions: list[np.ndarray] = []
    for frame in frames:
        pose = vo_result.poses.get(frame.image_id)
        if pose is None:
            continue
        positions.append(np.asarray(pose.t, dtype=np.float64) * scale)

    if len(positions) < 2:
        return []

    candidates: List[Tuple[np.ndarray, float, str]] = []
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    for idx in range(len(positions) - 1):
        p0 = positions[idx]
        p1 = positions[idx + 1]
        baseline = p1 - p0
        length = np.linalg.norm(baseline)
        if length < 1e-5:
            continue

        forward = baseline / length
        lateral = np.array([-forward[1], forward[0], 0.0], dtype=np.float64)
        if np.linalg.norm(lateral) < 1e-6:
            lateral = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        lateral /= np.linalg.norm(lateral)

        alphas = np.linspace(0.1, 0.9, 4, dtype=np.float64)
        offsets = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float64)
        base_uncert = 0.35 + 0.1 * (1.0 - np.clip(length / 8.0, 0.0, 1.0))

        for alpha in alphas:
            center = (1.0 - alpha) * p0 + alpha * p1
            for offset in offsets:
                point = center + lateral * float(offset) - up * ASSUMED_CAM_HEIGHT
                candidates.append((point.astype(np.float64), float(base_uncert), "vo"))

    return candidates


def _evaluate_candidates(
    seq_id: str,
    frames: Sequence[FrameMeta],
    centers: Mapping[str, np.ndarray],
    mask_priors: Mapping[str, float],
    candidates: Iterable[Tuple[np.ndarray, float, str]],
) -> List[GroundPoint]:
    results: List[GroundPoint] = []
    if not candidates:
        return results

    for point, base_uncert, method in candidates:
        gp = _classify_point(
            seq_id,
            point,
            method,
            frames,
            centers,
            mask_priors,
            base_uncertainty=base_uncert,
        )
        if gp is not None:
            results.append(gp)
    return results


def _classify_point(
    seq_id: str,
    point: np.ndarray,
    method: str,
    frames: Sequence[FrameMeta],
    centers: Mapping[str, np.ndarray],
    mask_priors: Mapping[str, float],
    base_uncertainty: float,
) -> GroundPoint | None:
    supports: List[Tuple[str, float, float]] = []
    ground_z = point[2]

    for frame in frames:
        center = centers.get(frame.image_id)
        if center is None:
            continue
        dz = ground_z - (center[2] - ASSUMED_CAM_HEIGHT)
        if abs(dz) > 1.8:
            continue
        horizontal = np.linalg.norm(point[:2] - center[:2])
        if horizontal > SUPPORT_RADIUS_M:
            continue
        weight = 1.0 - (horizontal / SUPPORT_RADIUS_M)
        prob = mask_priors.get(frame.image_id, 0.65)
        supports.append((frame.image_id, prob, max(0.1, weight)))

    if len(supports) < 2:
        return None

    weights = np.asarray([w for _, _, w in supports], dtype=np.float32)
    probs = np.asarray([p for _, p, _ in supports], dtype=np.float32)
    sem_prob = float(np.clip(np.average(probs, weights=weights), 0.0, 1.0))
    if sem_prob < 0.5:
        return None

    supports.sort(key=lambda item: item[2], reverse=True)
    image_ids = [item[0] for item in supports[:5]]
    tri_angle = _triangulation_angle(point, image_ids, centers)
    if tri_angle is not None and tri_angle < constants.MIN_TRIANG_ANGLE_DEG * 0.5:
        return None

    view_count = len(image_ids)
    uncertainty = _estimate_uncertainty(tri_angle, view_count, sem_prob, base_uncertainty)

    return GroundPoint(
        x=float(point[0]),
        y=float(point[1]),
        z=float(point[2]),
        method=str(method),
        seq_id=seq_id,
        image_ids=image_ids,
        view_count=view_count,
        sem_prob=sem_prob,
        tri_angle_deg=None if tri_angle is None else float(tri_angle),
        uncertainty_m=float(uncertainty),
    )


def _mask_prior(mask: np.ndarray | None) -> float:
    if mask is None:
        return 0.7
    rows = mask.shape[0]
    bottom = mask[int(rows * 0.6) :] if rows >= 4 else mask
    return float(np.clip(bottom.mean(), 0.0, 1.0))


def _load_mask(mask_dir: Path, image_id: str) -> np.ndarray | None:
    path = mask_dir / f"{image_id}.npz"
    if not path.exists():
        return None
    try:
        with np.load(path) as data:
            prob = np.asarray(data.get("prob"), dtype=np.float32)
            if prob.ndim != 2 or prob.size == 0:
                return None
            return np.clip(prob, 0.0, 1.0)
    except Exception:
        return None


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(vec[:2])
    if norm < 1e-6:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    vec = vec / norm
    return np.array([vec[0], vec[1], 0.0], dtype=np.float64)


def _triangulation_angle(
    point: np.ndarray,
    image_ids: Sequence[str],
    centers: Mapping[str, np.ndarray],
) -> float | None:
    if len(image_ids) < 2:
        return 0.0
    angles: List[float] = []
    for i in range(len(image_ids) - 1):
        c1 = centers.get(image_ids[i])
        if c1 is None:
            continue
        v1 = np.asarray(point, dtype=np.float64) - c1
        n1 = np.linalg.norm(v1)
        if n1 < 1e-6:
            continue
        for j in range(i + 1, len(image_ids)):
            c2 = centers.get(image_ids[j])
            if c2 is None:
                continue
            v2 = np.asarray(point, dtype=np.float64) - c2
            n2 = np.linalg.norm(v2)
            if n2 < 1e-6:
                continue
            cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            angles.append(float(np.degrees(np.arccos(cos_theta))))
    if not angles:
        return 0.0
    return float(np.median(angles))


def _estimate_uncertainty(
    tri_angle_deg: float | None,
    view_count: int,
    sem_prob: float,
    base_uncertainty: float,
) -> float:
    angle = max(tri_angle_deg or 0.0, 0.5)
    angle_term = 1.0 / max(math.tan(math.radians(angle)), 0.05)
    view_term = 1.0 / max(view_count, 1)
    sem_term = max(0.3, 1.2 - sem_prob)
    uncert = base_uncertainty * (0.6 + 0.4 * angle_term) * (0.7 + 0.5 * view_term) * sem_term
    return float(np.clip(uncert, 0.05, 0.6))
