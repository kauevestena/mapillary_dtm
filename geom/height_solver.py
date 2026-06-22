"""Per-sequence scale and camera-height estimation.

After Umeyama alignment of each reconstruction to GNSS ENU coordinates,
estimates a single camera-above-ground height ``h_cam`` per (sequence,
reconstruction) pair via iteratively reweighted Least Squares.

The physical constraint is that ``h_cam`` is constant within a sequence
(same car, same drive), with only small variations due to suspension
displacement (±3 cm RMS, ±10 cm outlier; see
documentation/extras/car_suspension_displacement.md).
"""
from __future__ import annotations

import logging
import numpy as np
from collections import defaultdict
from typing import Dict, Iterable, Mapping, Tuple

from .. import constants
from ..common_core import Anchor, FrameMeta, ReconstructionResult, Pose
from .utils import positions_from_frames, umeyama_alignment

logger = logging.getLogger(__name__)


def align_reconstruction_to_gnss(recon: ReconstructionResult, frames: list[FrameMeta]) -> float:
    """Aligns the reconstruction to GNSS ENU positions using Umeyama algorithm.
    Modifies the reconstruction in-place and returns the computed scale.
    """
    if len(frames) < 3 or not recon.poses:
        raise ValueError("Cannot align reconstruction to GNSS: Less than 3 frames or no valid poses.")

    gnss_pos, _ = positions_from_frames(frames)

    # Extract corresponding SfM poses
    sfm_pos = []
    valid_gnss_pos = []

    for i, frame in enumerate(frames):
        if frame.image_id in recon.poses:
            sfm_pos.append(recon.poses[frame.image_id].t)
            valid_gnss_pos.append(gnss_pos[i])

    if len(sfm_pos) < 3:
        raise ValueError(f"Cannot align reconstruction to GNSS: Less than 3 corresponding SfM poses found ({len(sfm_pos)}).")

    src = np.array(sfm_pos, dtype=np.float64)
    dst = np.array(valid_gnss_pos, dtype=np.float64)

    R, t, s = umeyama_alignment(src, dst)

    if not np.isfinite(s) or s < 1e-4 or s > 1000.0:
        raise ValueError(f"Invalid Umeyama scale {s:.3f} for {recon.seq_id}. Alignment failed.")

    # Apply alignment in-place
    for image_id, pose in recon.poses.items():
        aligned_t = s * (R @ pose.t) + t
        aligned_R = R @ pose.R
        recon.poses[image_id] = Pose(R=aligned_R, t=aligned_t)

    if recon.points_xyz is not None and recon.points_xyz.size > 0:
        # Align point cloud
        recon.points_xyz = s * (recon.points_xyz @ R.T) + t

    return float(s)


def _estimate_h_cam_ls(
    recon: ReconstructionResult,
    frames: list[FrameMeta],
    anchors: list[Anchor],
) -> float:
    """Estimate camera-above-ground height via iteratively reweighted LS.

    Collects per-frame observations of (camera_Z - ground_Z) using GNSS
    altitudes and anchor heights, then fits a single ``h_cam`` with
    iterative outlier rejection based on suspension displacement tolerance.

    Parameters
    ----------
    recon : ReconstructionResult
        Already Umeyama-aligned reconstruction (poses in ENU).
    frames : list[FrameMeta]
        Frame metadata with GNSS altitudes.
    anchors : list[Anchor]
        Ground control points for this sequence.

    Returns
    -------
    float
        Best-fit camera height above ground in metres.
    """
    observations: list[float] = []

    # Strategy 1: Use anchors if available.
    # Anchors provide (alt_ellip, height_m) — height_m is GPS antenna height,
    # and alt_ellip is the antenna altitude.  Ground altitude = alt_ellip - height_m.
    # Camera altitude comes from the aligned trajectory.
    if anchors:
        anchor_by_seq = {a.seq_id: a for a in anchors}
        for a in anchors:
            # Ground elevation at the anchor location
            ground_alt = a.alt_ellip - a.height_m
            # Find frames near this anchor to get camera Z
            for frame in frames:
                if frame.image_id not in recon.poses:
                    continue
                cam_z_enu = float(recon.poses[frame.image_id].t[2])
                # The camera Z in ENU is relative to the ENU origin,
                # and the GNSS alt is ellipsoidal.  After Umeyama alignment,
                # cam_z_enu ≈ gnss_z_enu for the same point.  We need the
                # difference to the ground.
                if frame.alt_ellip is not None and np.isfinite(frame.alt_ellip):
                    # h_cam ≈ frame.alt_ellip - ground_alt  (from metadata)
                    obs = frame.alt_ellip - ground_alt
                    if np.isfinite(obs) and constants.H_MIN_M <= obs <= constants.H_MAX_M:
                        observations.append(obs)

    # Strategy 2: Without anchors, we cannot isolate the camera-to-ground
    # height from GNSS data alone (GNSS altitude = ground_elevation + h_cam,
    # and we don't know ground_elevation independently).
    if not observations:
        raise ValueError(
            "Cannot estimate h_cam: No valid ground control anchors available to "
            "establish true ground elevation, and synthetic fallback is prohibited. "
            "Dataset must include anchors or be expanded to allow estimation."
        )

    obs = np.array(observations, dtype=np.float64)

    # Iteratively reweighted LS: estimate h_cam as the median,
    # then reject outliers beyond the suspension tolerance and re-estimate.
    for iteration in range(3):
        h_cam = float(np.median(obs))
        residuals = np.abs(obs - h_cam)
        inliers = residuals <= constants.SUSPENSION_OUTLIER_M
        if inliers.sum() < 1:
            break
        if inliers.all():
            # All inliers — refine with mean for LS optimality
            h_cam = float(np.mean(obs[inliers]))
            break
        obs = obs[inliers]

    # Final LS estimate from remaining inliers
    h_cam = float(np.mean(obs))

    # Clamp to physical bounds
    h_cam = max(constants.H_MIN_M, min(constants.H_MAX_M, h_cam))

    return h_cam


def solve_scale_and_h(
    reconA: Mapping[str, ReconstructionResult],
    reconB: Mapping[str, ReconstructionResult],
    vo: Mapping[str, ReconstructionResult],
    anchors: Iterable[Anchor],
    seqs: Mapping[str, Iterable[FrameMeta]],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute per-sequence scale factors and camera heights.

    This function:
    1. Aligns each reconstruction (OpenSfM, COLMAP, VO) to GNSS ENU
       using a 3D Umeyama similarity transform (in-place).
    2. Estimates a single camera-above-ground height ``h_cam`` per sequence
       via iteratively reweighted Least Squares, constrained by the
       physical suspension displacement tolerance.

    After alignment, the returned ``scales`` are all 1.0 (the alignment
    absorbs the scale).  The ``heights`` dict contains the best-fit
    ``h_cam`` per sequence.

    Args:
        reconA: First reconstruction result mapping (e.g., OpenSfM)
        reconB: Second reconstruction result mapping (e.g., COLMAP)
        vo: Visual odometry reconstruction result mapping
        anchors: Ground control points or reference anchors
        seqs: Frame metadata organized by sequence ID

    Returns:
        Tuple of (scales, heights) dictionaries keyed by sequence ID

    Raises:
        ValueError: If constraints fail (e.g., all sequences lack valid GPS data)
    """

    anchors_list = list(anchors)
    anchors_by_seq: Dict[str, list[Anchor]] = defaultdict(list)
    for anchor in anchors_list:
        anchors_by_seq[anchor.seq_id].append(anchor)

    scales: Dict[str, float] = {}
    heights: Dict[str, float] = {}

    # Track sequences with issues for diagnostic logging
    no_frames = []
    insufficient_gnss = []
    no_recon_data = []

    for seq_id, frames in seqs.items():
        frames_list = list(frames)
        if not frames_list:
            no_frames.append(seq_id)
            continue

        try:
            gnss_avg = _average_step_from_gnss(frames_list)
        except ValueError as e:
            logger.warning(f"Sequence {seq_id}: {e} Skipping sequence.")
            continue

        aligned_any = False
        h_cam_estimates: list[float] = []

        # Align each reconstruction in-place and estimate h_cam
        for recon_name, recon_map in [("reconA", reconA), ("reconB", reconB), ("vo", vo)]:
            recon = recon_map.get(seq_id)
            if not recon:
                continue

            try:
                s = align_reconstruction_to_gnss(recon, frames_list)
            except ValueError as e:
                logger.warning(f"Sequence {seq_id} / {recon_name}: {e} Skipping reconstruction.")
                continue

            aligned_any = True

            # Estimate h_cam for this (sequence, reconstruction) pair
            seq_anchors = anchors_by_seq.get(seq_id, [])
            try:
                h = _estimate_h_cam_ls(recon, frames_list, seq_anchors)
                h_cam_estimates.append(h)
                logger.info(
                    f"Sequence {seq_id} / {recon_name}: "
                    f"Umeyama scale={s:.4f}, h_cam={h:.3f} m"
                )
            except ValueError as e:
                logger.warning(f"Sequence {seq_id} / {recon_name}: {e} Skipping height estimation for this recon.")

        if not aligned_any:
            no_recon_data.append(seq_id)
            continue

        if not h_cam_estimates:
            logger.warning(f"Sequence {seq_id}: Could not estimate h_cam from any reconstruction. Synthetic fallback is prohibited. Skipping sequence.")
            continue

        # Poses are now aligned in-place → scale = 1.0
        scales[seq_id] = 1.0

        # Best h_cam for this sequence: median across reconstructions
        heights[seq_id] = float(np.median(h_cam_estimates))

        logger.info(
            f"Sequence {seq_id}: final h_cam={heights[seq_id]:.3f} m "
            f"(from {len(h_cam_estimates)} reconstruction(s), "
            f"suspension tolerance ±{constants.SUSPENSION_OUTLIER_M*100:.0f} cm)"
        )

    # Summary diagnostics
    if no_frames:
        logger.warning(f"Sequences with no frames: {', '.join(no_frames)}")
    if insufficient_gnss:
        logger.warning(
            f"Sequences with insufficient GPS coverage (avg step < 1e-6m): {', '.join(insufficient_gnss)}. "
            "Check GNSS quality or consider using reference tracks."
        )
    if no_recon_data:
        logger.info(
            f"Sequences without valid reconstruction data: {', '.join(no_recon_data)}. "
            "This may indicate reconstruction failures or insufficient image overlap."
        )

    # Raise if no sequences were successfully processed
    if not scales:
        raise ValueError(
            "Failed to compute scale for any sequence. "
            "Possible causes: (1) Insufficient GPS data, (2) Reconstruction failures, "
            "(3) Extreme discrepancies between GNSS and reconstruction. "
            "Check input data quality and reconstruction outputs."
        )

    return scales, heights


def _average_step_from_gnss(frames: Iterable[FrameMeta]) -> float:
    """Calculate average step size from GNSS positions.

    Returns:
        Average step size in meters

    Raises:
        ValueError: If there's insufficient data to calculate an average step
    """
    frames = list(frames)
    if len(frames) < 2:
        raise ValueError("Cannot calculate GNSS step: Less than 2 frames available.")

    try:
        positions, _ = positions_from_frames(frames)
        if positions.shape[0] < 2:
            raise ValueError("Cannot calculate GNSS step: Less than 2 valid GNSS positions.")
        diffs = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        if diffs.size == 0:
            raise ValueError("Cannot calculate GNSS step: No valid differences between positions.")

        # Filter out non-finite values
        diffs = diffs[np.isfinite(diffs)]
        if diffs.size == 0:
            raise ValueError("Cannot calculate GNSS step: No finite differences between positions.")

        avg_step = float(np.mean(diffs))
        if not np.isfinite(avg_step):
            raise ValueError("Cannot calculate GNSS step: Result is not finite.")
        return avg_step
    except (ValueError, RuntimeError) as e:
        logger.warning(f"Error computing GNSS step: {e}")
        raise ValueError(f"Failed to calculate average step from GNSS: {e}")


def _average_step_from_recon(recon: ReconstructionResult) -> float:
    """Calculate average step size from reconstruction poses.

    Returns:
        Average step size in reconstruction units

    Raises:
        ValueError: If there's insufficient data to calculate an average step
    """
    frames = recon.frames
    if len(frames) < 2:
        raise ValueError("Cannot calculate reconstruction step: Less than 2 frames available.")

    try:
        translations = np.array([recon.poses[f.image_id].t for f in frames])
        diffs = np.linalg.norm(np.diff(translations, axis=0), axis=1)
        if diffs.size == 0:
            raise ValueError("Cannot calculate reconstruction step: No valid differences between poses.")

        # Filter out non-finite values
        diffs = diffs[np.isfinite(diffs)]
        if diffs.size == 0:
            raise ValueError("Cannot calculate reconstruction step: No finite differences between poses.")

        avg_step = float(np.mean(diffs))
        if not np.isfinite(avg_step):
            raise ValueError("Cannot calculate reconstruction step: Result is not finite.")
        return avg_step
    except (KeyError, ValueError, RuntimeError) as e:
        logger.warning(f"Error computing reconstruction step: {e}")
        raise ValueError(f"Failed to calculate average step from reconstruction: {e}")


def estimate_h_cam_from_dtm(
    recon: Mapping[str, ReconstructionResult],
    scales: Mapping[str, float],
    dtm: np.ndarray,
    grid: "GridSpec",
) -> Dict[str, float]:
    """Estimate h_cam per sequence by sampling the DTM under each camera.

    This is the **second pass** estimator: after the DTM has been built
    (with a default h_cam), we sample the DTM elevation directly under
    each aligned camera position and compute::

        h_cam_obs(i) = camera_Z(i) - DTM_Z(i)

    Then we fit a single ``h_cam`` per sequence using iterative median +
    outlier rejection bounded by the suspension displacement tolerance.

    Parameters
    ----------
    recon : Mapping[str, ReconstructionResult]
        Reconstructions (already Umeyama-aligned, poses in ENU).
    scales : Mapping[str, float]
        Per-sequence scale factors (typically 1.0 after alignment).
    dtm : np.ndarray
        2-D array of ground elevations in ENU-Z.
    grid : GridSpec
        Grid specification (ix_min, iy_min, width, height, res).

    Returns
    -------
    Dict[str, float]
        Per-sequence best-fit camera height above ground.
    """
    from math import floor

    heights: Dict[str, float] = {}

    for seq_id, result in recon.items():
        if not result.poses:
            continue

        scale = float(scales.get(seq_id, 1.0))
        observations: list[float] = []

        for img_id, pose in result.poses.items():
            cam_x, cam_y, cam_z = pose.t * scale

            # Map to grid indices
            ix = int(floor(cam_x / grid.res)) - grid.ix_min
            iy = int(floor(cam_y / grid.res)) - grid.iy_min

            if not (0 <= ix < grid.width and 0 <= iy < grid.height):
                continue

            dtm_z = float(dtm[iy, ix])
            if not np.isfinite(dtm_z):
                continue

            obs = float(cam_z) - dtm_z
            if np.isfinite(obs) and obs > 0:
                observations.append(obs)

        if not observations:
            raise ValueError(
                f"Sequence {seq_id}: no valid DTM samples under cameras. "
                "Cannot estimate h_cam, and synthetic fallback is prohibited. "
                "Dataset must be expanded or parameters modified."
            )

        obs_arr = np.array(observations, dtype=np.float64)

        # Iterative median + outlier rejection.
        # Note: we use a wider tolerance than SUSPENSION_OUTLIER_M here
        # because the DTM grid cells carry their own reconstruction noise
        # (~0.3-0.5 m) on top of the physical suspension variation.
        dtm_outlier_tol = 0.5  # metres
        for _iteration in range(3):
            h_cam = float(np.median(obs_arr))
            residuals = np.abs(obs_arr - h_cam)
            inliers = residuals <= dtm_outlier_tol
            if inliers.sum() < 1:
                break
            if inliers.all():
                h_cam = float(np.mean(obs_arr[inliers]))
                break
            obs_arr = obs_arr[inliers]

        h_cam = float(np.mean(obs_arr))

        # Clamp to physical bounds
        h_cam = max(constants.H_MIN_M, min(constants.H_MAX_M, h_cam))

        n_obs = len(observations)
        n_inliers = len(obs_arr)
        logger.info(
            "Sequence %s: h_cam=%.3f m from DTM "
            "(%d observations, %d inliers after outlier rejection)",
            seq_id, h_cam, n_obs, n_inliers,
        )
        heights[seq_id] = h_cam

    return heights

