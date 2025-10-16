"""Visual odometry module with OpenCV-backed track estimation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .. import constants
from ..common_core import FrameMeta, Pose, ReconstructionResult
from ..ingest.image_loader import ImageryLoader
from .utils import heading_matrix, positions_from_frames

try:  # Optional dependency for real VO pipeline
    import cv2  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    cv2 = None

log = logging.getLogger(__name__)


def run(
    seqs: Mapping[str, List[FrameMeta]],
    rng_seed: int = 3025,
    *,
    imagery_root: Optional[Path | str] = None,
    force_synthetic: bool = False,
    min_inliers: Optional[int] = None,
) -> Dict[str, ReconstructionResult]:
    """
    Estimate relative trajectories per sequence using visual odometry.

    When OpenCV and cached imagery are available, an ORB + Essential-matrix
    pipeline is used to recover relative motion. Otherwise the routine falls
    back to the deterministic synthetic path used by earlier milestones.
    """

    if not seqs:
        return {}

    min_inliers = max(
        12,
        int(min_inliers or constants.VO_MIN_INLIERS),
    )

    if force_synthetic or cv2 is None:
        if force_synthetic:
            log.info("VO forced to synthetic mode (flag or CLI)")
        elif cv2 is None:
            log.info("OpenCV not available; using synthetic VO path")
        return _run_synthetic(seqs, rng_seed=rng_seed)

    results: Dict[str, ReconstructionResult] = {}
    loader = ImageryLoader(imagery_root)
    used_synthetic = False

    for seq_id, frames in seqs.items():
        if not frames:
            continue

        track = _run_opencv_sequence(seq_id, frames, loader, min_inliers=min_inliers)
        if track is None:
            used_synthetic = True
            syn = _run_synthetic({seq_id: frames}, rng_seed=rng_seed)
            if syn:
                results.update(syn)
            continue

        results[seq_id] = track

    if used_synthetic:
        log.info("VO fallback to synthetic path for some sequences (missing imagery or low match count)")

    return results


def _run_opencv_sequence(
    seq_id: str,
    frames: Sequence[FrameMeta],
    loader: ImageryLoader,
    *,
    min_inliers: int,
) -> Optional[ReconstructionResult]:
    images: List[tuple[FrameMeta, np.ndarray]] = []
    for frame in frames:
        image = loader.load_gray(frame)
        if image is None:
            log.debug("VO imagery missing for %s/%s", seq_id, frame.image_id)
            continue
        images.append((frame, image))

    if len(images) < 2:
        log.info("VO sequence %s has insufficient imagery (%d frames)", seq_id, len(images))
        return None

    poses: Dict[str, Pose] = {}
    centers: Dict[str, np.ndarray] = {}
    orientations: Dict[str, np.ndarray] = {}

    current_center = np.zeros(3, dtype=np.float64)
    current_R = np.eye(3, dtype=np.float64)

    first_frame, _ = images[0]
    poses[first_frame.image_id] = Pose(R=current_R.copy(), t=current_center.copy())
    centers[first_frame.image_id] = current_center.copy()
    orientations[first_frame.image_id] = current_R.copy()

    orb = cv2.ORB_create(constants.VO_ORB_FEATURES)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    total_pairs = 0
    total_matches = 0
    inlier_counts: List[int] = []
    step_lengths: List[float] = []

    for idx in range(len(images) - 1):
        frame_a, img_a = images[idx]
        frame_b, img_b = images[idx + 1]
        K = _camera_matrix(frame_a)
        if K is None:
            log.debug("VO missing intrinsics for %s; falling back to synthetic step", frame_a.image_id)
            delta_center, current_R = _synthetic_step(current_center, current_R, frame_a, frame_b)
            current_center = current_center + delta_center
            step_lengths.append(float(np.linalg.norm(delta_center)))
            total_pairs += 1
            poses[frame_b.image_id] = Pose(R=current_R.copy(), t=current_center.copy())
            centers[frame_b.image_id] = current_center.copy()
            orientations[frame_b.image_id] = current_R.copy()
            continue

        kp1, des1 = orb.detectAndCompute(img_a, None)
        kp2, des2 = orb.detectAndCompute(img_b, None)
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            log.debug("VO insufficient keypoints between %s and %s", frame_a.image_id, frame_b.image_id)
            delta_center, current_R = _synthetic_step(current_center, current_R, frame_a, frame_b)
            current_center = current_center + delta_center
            step_lengths.append(float(np.linalg.norm(delta_center)))
            total_pairs += 1
            poses[frame_b.image_id] = Pose(R=current_R.copy(), t=current_center.copy())
            centers[frame_b.image_id] = current_center.copy()
            orientations[frame_b.image_id] = current_R.copy()
            continue

        if constants.VO_USE_RATIO_TEST:
            matches_knn = bf.knnMatch(des1, des2, k=2)
            matches = []
            for m, n in matches_knn:
                if m.distance < constants.VO_RATIO_TEST * n.distance:
                    matches.append(m)
        else:
            matches = bf.match(des1, des2)

        if len(matches) < min_inliers:
            log.debug("VO match count below threshold (%d < %d) between %s and %s", len(matches), min_inliers, frame_a.image_id, frame_b.image_id)
            delta_center, current_R = _synthetic_step(current_center, current_R, frame_a, frame_b)
            current_center = current_center + delta_center
            step_lengths.append(float(np.linalg.norm(delta_center)))
            total_pairs += 1
            poses[frame_b.image_id] = Pose(R=current_R.copy(), t=current_center.copy())
            centers[frame_b.image_id] = current_center.copy()
            orientations[frame_b.image_id] = current_R.copy()
            continue

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        E, mask = cv2.findEssentialMat(
            pts1,
            pts2,
            K,
            method=cv2.RANSAC,
            threshold=constants.VO_RANSAC_THRESH,
            prob=0.999,
        )
        if E is None or E.size == 0:
            log.debug("VO essential matrix failed between %s and %s", frame_a.image_id, frame_b.image_id)
            delta_center, current_R = _synthetic_step(current_center, current_R, frame_a, frame_b)
            current_center = current_center + delta_center
            step_lengths.append(float(np.linalg.norm(delta_center)))
            total_pairs += 1
            poses[frame_b.image_id] = Pose(R=current_R.copy(), t=current_center.copy())
            centers[frame_b.image_id] = current_center.copy()
            orientations[frame_b.image_id] = current_R.copy()
            continue

        _, R_rel, t_rel, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
        inliers = int(mask_pose.sum()) if mask_pose is not None else 0
        if inliers < min_inliers:
            log.debug("VO recoverPose inliers below threshold (%d < %d)", inliers, min_inliers)
            delta_center, current_R = _synthetic_step(current_center, current_R, frame_a, frame_b)
            current_center = current_center + delta_center
            step_lengths.append(float(np.linalg.norm(delta_center)))
            total_pairs += 1
            poses[frame_b.image_id] = Pose(R=current_R.copy(), t=current_center.copy())
            centers[frame_b.image_id] = current_center.copy()
            orientations[frame_b.image_id] = current_R.copy()
            continue

        delta_center = _propagate_delta(current_R, t_rel)
        current_center = current_center + delta_center
        current_R = R_rel @ current_R

        step_lengths.append(float(np.linalg.norm(delta_center)))
        total_matches += len(matches)
        inlier_counts.append(inliers)
        total_pairs += 1

        poses[frame_b.image_id] = Pose(R=current_R.copy(), t=current_center.copy())
        centers[frame_b.image_id] = current_center.copy()
        orientations[frame_b.image_id] = current_R.copy()

    if len(poses) < 2:
        return None

    frames_sorted = [frame for frame in frames if frame.image_id in poses]
    step_mean = float(np.mean(step_lengths)) if step_lengths else 1.0
    metadata = {
        "scale": max(step_mean, 1e-6),
        "mode": "opencv",
        "pairs_processed": total_pairs,
        "avg_inliers": float(np.mean(inlier_counts)) if inlier_counts else 0.0,
        "avg_matches": float(total_matches / max(total_pairs, 1)),
    }

    return ReconstructionResult(
        seq_id=seq_id,
        frames=frames_sorted,
        poses=poses,
        points_xyz=np.zeros((0, 3), dtype=np.float32),
        source="vo",
        metadata=metadata,
    )


def _propagate_delta(R_world: np.ndarray, t_rel: np.ndarray) -> np.ndarray:
    vec = np.asarray(t_rel, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        norm = 1.0
    vec = vec / norm
    delta = R_world.T @ vec
    if not np.isfinite(delta).all():
        delta = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return delta


def _synthetic_step(
    current_center: np.ndarray,
    current_R: np.ndarray,
    frame_a: FrameMeta,
    frame_b: FrameMeta,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback when real VO fails for a frame pair."""
    del frame_a, frame_b  # parameters kept for signature clarity
    positions = np.vstack([current_center, current_center + np.array([1.0, 0.0, 0.0], dtype=np.float64)])
    baseline = heading_matrix(positions, 0)
    delta = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    R_next = baseline @ current_R
    return delta, R_next


def _camera_matrix(frame: FrameMeta) -> Optional[np.ndarray]:
    params = frame.cam_params or {}
    width = float(params.get("width") or params.get("image_width") or 2048.0)
    height = float(params.get("height") or params.get("image_height") or 1536.0)
    fx = params.get("fx_px")
    fy = params.get("fy_px")

    if fx is None or fy is None:
        focal_norm = params.get("focal") or params.get("f") or 1.0
        scale_px = max(width, height)
        fx = fy = float(focal_norm) * float(scale_px)

    cx = params.get("cx_px")
    cy = params.get("cy_px")
    if cx is None or cy is None:
        pp = params.get("principal_point")
        if isinstance(pp, (list, tuple)) and len(pp) == 2:
            cx = float(pp[0]) * width
            cy = float(pp[1]) * height
        else:
            cx = width * 0.5
            cy = height * 0.5

    matrix = np.array(
        [
            [float(fx), 0.0, float(cx)],
            [0.0, float(fy), float(cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    if not np.isfinite(matrix).all():
        return None
    return matrix


def _run_synthetic(
    seqs: Mapping[str, List[FrameMeta]],
    rng_seed: int,
) -> Dict[str, ReconstructionResult]:
    rng = np.random.default_rng(rng_seed)
    results: Dict[str, ReconstructionResult] = {}

    for seq_id, frames in seqs.items():
        if not frames:
            continue

        positions, _ = positions_from_frames(frames)
        if positions.size == 0:
            continue

        positions -= positions[0]
        step_norms = (
            np.linalg.norm(np.diff(positions, axis=0), axis=1)
            if positions.shape[0] > 1
            else np.array([1.0])
        )
        scale = float(np.mean(step_norms)) if step_norms.size else 1.0
        if scale <= 1e-6:
            scale = 1.0
        rel_positions = positions / scale

        poses = {}
        for idx, frame in enumerate(frames):
            pos = rel_positions[idx] + rng.normal(scale=0.01, size=3)
            R = heading_matrix(rel_positions, idx)
            poses[frame.image_id] = Pose(R=R, t=pos)

        results[seq_id] = ReconstructionResult(
            seq_id=seq_id,
            frames=list(frames),
            poses=poses,
            points_xyz=np.zeros((0, 3), dtype=np.float32),
            source="vo",
            metadata={
                "scale": scale,
                "rng_seed": rng_seed,
                "mode": "synthetic",
            },
        )

    return results
