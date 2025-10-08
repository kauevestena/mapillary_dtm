"""Synthetic OpenSfM reconstruction scaffolding with optional self-calibration."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np

from ..common_core import FrameMeta, Pose, ReconstructionResult
from .utils import heading_matrix, positions_from_frames, synthetic_ground_offsets

logger = logging.getLogger(__name__)


def _extract_correspondences_for_frame(
    frame: FrameMeta,
    pose: Pose,
    points_xyz: np.ndarray,
    rng: np.random.Generator,
    max_points: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract synthetic 3D-2D correspondences for a single frame.

    In a real implementation, this would query the SfM reconstruction
    for 3D points visible in the frame and their 2D observations.
    Here we simulate this by projecting 3D points into the camera.

    Args:
        frame: Camera frame metadata
        pose: Camera pose (R, t)
        points_xyz: 3D points from reconstruction (N, 3)
        rng: Random number generator
        max_points: Maximum correspondences to extract

    Returns:
        points_3d: (M, 3) array of 3D world points
        points_2d: (M, 2) array of normalized 2D projections
    """
    if points_xyz.shape[0] == 0:
        return np.zeros((0, 3)), np.zeros((0, 2))

    # Transform points to camera frame
    R_inv = pose.R.T
    points_cam = (points_xyz - pose.t) @ R_inv.T

    # Keep points in front of camera (positive Z)
    valid = points_cam[:, 2] > 0.1
    points_cam = points_cam[valid]

    if points_cam.shape[0] == 0:
        return np.zeros((0, 3)), np.zeros((0, 2))

    # Limit to max_points
    if points_cam.shape[0] > max_points:
        indices = rng.choice(points_cam.shape[0], max_points, replace=False)
        points_cam = points_cam[indices]
        valid_indices = np.where(valid)[0][indices]
    else:
        valid_indices = np.where(valid)[0]

    # Project to normalized image coordinates (simple pinhole)
    # In normalized coords: x = X/Z, y = Y/Z
    points_2d_norm = points_cam[:, :2] / points_cam[:, 2:3]

    # Add small projection noise to simulate observation error
    noise = rng.normal(
        scale=0.002, size=points_2d_norm.shape
    )  # ~0.2% normalized coords
    points_2d_norm += noise

    # Get corresponding 3D world points
    points_3d_world = points_xyz[valid_indices]

    return points_3d_world, points_2d_norm


def _refine_sequence_cameras(
    frames: List[FrameMeta],
    poses: Dict[str, Pose],
    points_xyz: np.ndarray,
    method: str = "full",
    rng: Optional[np.random.Generator] = None,
) -> tuple[List[FrameMeta], Dict]:
    """
    Apply self-calibration to all cameras in a sequence.

    Args:
        frames: List of FrameMeta with camera parameters
        poses: Dict of image_id -> Pose
        points_xyz: Reconstructed 3D points
        method: 'full' for complete refinement, 'quick' for fast path
        rng: Random number generator for synthetic correspondence extraction

    Returns:
        refined_frames: List of FrameMeta with refined camera parameters
        metadata: Dictionary with refinement statistics
    """
    if rng is None:
        rng = np.random.default_rng()

    # Import self-calibration module
    try:
        from .self_calibration import refine_sequence_cameras
    except ImportError as e:
        logger.warning(f"Self-calibration module not available: {e}")
        return frames, {"error": "self_calibration not available"}

    # Build separate dictionaries for each input type
    sequence_data = {}  # image_id -> camera
    correspondences = {}  # image_id -> (points_3d, points_2d)
    pose_dicts = {}  # image_id -> pose matrix (4x4)
    image_sizes = {}  # image_id -> (width, height)

    for frame in frames:
        if frame.image_id not in poses:
            continue

        pose = poses[frame.image_id]

        # Extract correspondences
        points_3d, points_2d = _extract_correspondences_for_frame(
            frame, pose, points_xyz, rng
        )

        if points_3d.shape[0] < 10:
            logger.debug(
                f"Frame {frame.image_id}: Insufficient correspondences ({points_3d.shape[0]})"
            )
            continue

        # Build camera dict from FrameMeta.cam_params
        camera = _camera_from_frame(frame)

        # Get image size
        width = frame.cam_params.get("width", 4000)
        height = frame.cam_params.get("height", 3000)

        # Convert Pose to 4x4 matrix
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = pose.R
        pose_matrix[:3, 3] = pose.t

        # Store in dictionaries
        sequence_data[frame.image_id] = camera
        correspondences[frame.image_id] = (points_3d, points_2d)
        pose_dicts[frame.image_id] = pose_matrix
        image_sizes[frame.image_id] = (width, height)

    if not sequence_data:
        return frames, {"error": "No valid correspondences", "refined_count": 0}

    # Call self-calibration workflow
    try:
        results = refine_sequence_cameras(
            sequence_data, correspondences, pose_dicts, image_sizes, method=method
        )
    except Exception as e:
        logger.error(f"Self-calibration failed: {e}")
        return frames, {"error": str(e), "refined_count": 0}

    # Update frames with refined cameras
    refined_frames = []
    refined_count = 0
    total_improvement = 0.0

    for frame in frames:
        if frame.image_id in results:
            result = results[frame.image_id]
            refined_camera = result.refined_camera

            # Update cam_params with refined values
            new_cam_params = frame.cam_params.copy()
            new_cam_params["focal"] = refined_camera["focal"]
            new_cam_params["principal_point"] = refined_camera["principal_point"]

            # Update distortion if present
            for k in ["k1", "k2", "k3", "p1", "p2"]:
                if k in refined_camera:
                    new_cam_params[k] = refined_camera[k]

            # Create new FrameMeta with updated camera
            refined_frame = FrameMeta(
                image_id=frame.image_id,
                seq_id=frame.seq_id,
                captured_at_ms=frame.captured_at_ms,
                lon=frame.lon,
                lat=frame.lat,
                alt_ellip=frame.alt_ellip,
                camera_type=frame.camera_type,
                cam_params=new_cam_params,
                quality_score=frame.quality_score,
            )
            refined_frames.append(refined_frame)
            refined_count += 1
            total_improvement += result.improvement
        else:
            refined_frames.append(frame)

    # Compute statistics
    avg_improvement = total_improvement / max(refined_count, 1)

    metadata = {
        "refined_count": refined_count,
        "total_frames": len(frames),
        "avg_improvement_px": float(avg_improvement),
        "method": method,
    }

    return refined_frames, metadata


def _camera_from_frame(frame: FrameMeta) -> Dict:
    """
    Convert FrameMeta.cam_params to self-calibration camera format.

    Args:
        frame: FrameMeta with camera parameters

    Returns:
        Camera dictionary with focal, principal_point, distortion, projection_type
    """
    cam_params = frame.cam_params

    # Extract or set defaults
    focal = cam_params.get("focal", cam_params.get("f", 1.0))
    cx = cam_params.get(
        "cx",
        (
            cam_params.get("principal_point", [0.5, 0.5])[0]
            if isinstance(cam_params.get("principal_point"), list)
            else 0.5
        ),
    )
    cy = cam_params.get(
        "cy",
        (
            cam_params.get("principal_point", [0.5, 0.5])[1]
            if isinstance(cam_params.get("principal_point"), list)
            else 0.5
        ),
    )

    camera = {
        "focal": float(focal),
        "principal_point": [float(cx), float(cy)],
        "projection_type": (
            frame.camera_type
            if frame.camera_type in ["perspective", "fisheye", "spherical"]
            else "perspective"
        ),
    }

    # Add distortion parameters if present
    for k in ["k1", "k2", "k3", "p1", "p2"]:
        if k in cam_params:
            camera[k] = float(cam_params[k])

    return camera


def run(
    seqs: Mapping[str, List[FrameMeta]],
    rng_seed: int = 2025,
    refine_cameras: bool = False,
    refinement_method: str = "full",
) -> Dict[str, ReconstructionResult]:
    """
    Produce lightweight pseudo-reconstructions for testing/integration.

    Args:
        seqs: Mapping of sequence_id -> list of FrameMeta
        rng_seed: Random seed for reproducibility
        refine_cameras: If True, apply self-calibration to camera parameters
        refinement_method: 'full' for complete refinement, 'quick' for fast path

    Returns:
        Dictionary of sequence_id -> ReconstructionResult with refined cameras (if enabled)
    """

    rng = np.random.default_rng(rng_seed)
    results: Dict[str, ReconstructionResult] = {}

    for seq_id, frames in seqs.items():
        if not frames:
            continue

        positions, _ = positions_from_frames(frames)
        if positions.size == 0:
            continue

        poses = {}
        points: List[np.ndarray] = []
        offsets = synthetic_ground_offsets()

        for idx, frame in enumerate(frames):
            base_pos = positions[idx]
            noise = rng.normal(scale=0.05, size=3)
            pos = base_pos + noise
            R = heading_matrix(positions, idx)
            poses[frame.image_id] = Pose(R=R, t=pos)

            # Generate points in camera coordinates, then transform to world
            # This ensures points are in front of the camera
            for offset in offsets:
                # Transform offset from camera coords to world coords
                # In camera coords: X=right, Y=down, Z=forward
                # offset is in world coords (X,Y,Z), treating Z as vertical
                # We want points ahead and below the camera
                offset_cam = np.array(
                    [offset[0], offset[2], 5.0]
                )  # X, Y=down, Z=forward (5m ahead)
                offset_world = R @ offset_cam  # Transform to world
                jitter = rng.normal(scale=0.08, size=3)
                points.append(pos + offset_world + jitter)

        points_xyz = np.vstack(points) if points else np.zeros((0, 3), dtype=float)

        # Store initial reconstruction
        metadata = {
            "rng_seed": rng_seed,
            "point_count": int(points_xyz.shape[0]),
            "cameras_refined": False,
        }

        # Apply self-calibration if requested
        refined_frames = frames
        if refine_cameras and len(frames) > 0 and points_xyz.shape[0] >= 20:
            try:
                refined_frames, refine_meta = _refine_sequence_cameras(
                    frames=frames,
                    poses=poses,
                    points_xyz=points_xyz,
                    method=refinement_method,
                    rng=rng,
                )
                metadata.update(refine_meta)
                metadata["cameras_refined"] = True
                logger.info(
                    f"OpenSfM sequence {seq_id}: Camera refinement successful "
                    f"({refine_meta.get('refined_count', 0)}/{len(frames)} cameras)"
                )
            except Exception as e:
                logger.warning(
                    f"OpenSfM sequence {seq_id}: Camera refinement failed: {e}"
                )
                refined_frames = frames  # Fall back to original

        results[seq_id] = ReconstructionResult(
            seq_id=seq_id,
            frames=list(refined_frames),
            poses=poses,
            points_xyz=points_xyz.astype(np.float32),
            source="opensfm",
            metadata=metadata,
        )

    return results
