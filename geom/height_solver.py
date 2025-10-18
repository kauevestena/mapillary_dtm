"""Per-sequence scale and camera-height estimation."""
from __future__ import annotations

import logging
import numpy as np
from collections import defaultdict
from typing import Dict, Iterable, Mapping, Tuple

from .. import constants
from ..common_core import Anchor, FrameMeta, ReconstructionResult
from .utils import positions_from_frames

logger = logging.getLogger(__name__)


def solve_scale_and_h(
    reconA: Mapping[str, ReconstructionResult],
    reconB: Mapping[str, ReconstructionResult],
    vo: Mapping[str, ReconstructionResult],
    anchors: Iterable[Anchor],
    seqs: Mapping[str, Iterable[FrameMeta]],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute per-sequence scale factors and camera heights.
    
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

    anchors_by_seq = defaultdict(list)
    for anchor in anchors:
        anchors_by_seq[anchor.seq_id].append(anchor)

    scales: Dict[str, float] = {}
    heights: Dict[str, float] = {}
    
    # Track sequences with issues for diagnostic logging
    no_frames = []
    insufficient_gnss = []
    no_recon_data = []
    clamped_scales = []

    for seq_id, frames in seqs.items():
        frames_list = list(frames)
        if not frames_list:
            no_frames.append(seq_id)
            continue

        gnss_avg = _average_step_from_gnss(frames_list)
        
        # Check for numerical stability issues
        if not np.isfinite(gnss_avg):
            logger.warning(f"Sequence {seq_id}: Non-finite GNSS step detected, using default scale")
            gnss_avg = 1.0

        scale_candidates = []
        recon_sources = []
        for recon_name, recon_map in [("reconA", reconA), ("reconB", reconB), ("vo", vo)]:
            recon = recon_map.get(seq_id)
            if not recon:
                continue
            step = _average_step_from_recon(recon)
            
            # Numerical stability check
            if not np.isfinite(step):
                logger.warning(f"Sequence {seq_id}: Non-finite reconstruction step in {recon_name}")
                continue
                
            if step > 1e-6 and gnss_avg > 1e-6:
                candidate_scale = gnss_avg / step
                # Check for extreme scale values before averaging
                if not np.isfinite(candidate_scale):
                    logger.warning(f"Sequence {seq_id}: Non-finite scale from {recon_name}")
                    continue
                if candidate_scale < 0.01 or candidate_scale > 100.0:
                    logger.warning(
                        f"Sequence {seq_id}: Extreme scale {candidate_scale:.2f} from {recon_name} "
                        f"(GNSS step: {gnss_avg:.3f}m, recon step: {step:.3f})"
                    )
                    continue
                scale_candidates.append(candidate_scale)
                recon_sources.append(recon_name)
        
        if not scale_candidates:
            no_recon_data.append(seq_id)
            if gnss_avg < 1e-6:
                insufficient_gnss.append(seq_id)
                logger.warning(
                    f"Sequence {seq_id}: Insufficient GNSS data (avg step: {gnss_avg:.6f}m), "
                    "using default scale 1.0"
                )

        scale = float(np.mean(scale_candidates)) if scale_candidates else 1.0
        
        # Clamp to reasonable bounds and log if clamping occurs
        original_scale = scale
        scale = max(0.25, min(4.0, scale))
        if abs(scale - original_scale) > 1e-6:
            clamped_scales.append((seq_id, original_scale, scale))
            logger.info(
                f"Sequence {seq_id}: Scale clamped from {original_scale:.2f} to {scale:.2f} "
                f"(sources: {', '.join(recon_sources) if recon_sources else 'none'})"
            )
        
        scales[seq_id] = scale

        anchors_list = anchors_by_seq.get(seq_id, [])
        height = _height_from_anchors(frames_list, anchors_list)
        
        # Clamp height and log
        original_height = height
        heights[seq_id] = max(constants.H_MIN_M, min(constants.H_MAX_M, height))
        if abs(heights[seq_id] - original_height) > 1e-6:
            logger.info(
                f"Sequence {seq_id}: Height clamped from {original_height:.2f}m to {heights[seq_id]:.2f}m "
                f"(anchors: {len(anchors_list)})"
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
        Average step size in meters, or 1.0 if insufficient data
    """
    frames = list(frames)
    if len(frames) < 2:
        return 1.0
    
    try:
        positions, _ = positions_from_frames(frames)
        if positions.shape[0] < 2:
            return 1.0
        diffs = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        if diffs.size == 0:
            return 1.0
        
        # Filter out non-finite values
        diffs = diffs[np.isfinite(diffs)]
        if diffs.size == 0:
            return 1.0
            
        avg_step = float(np.mean(diffs))
        return avg_step if np.isfinite(avg_step) else 1.0
    except (ValueError, RuntimeError) as e:
        logger.warning(f"Error computing GNSS step: {e}")
        return 1.0


def _average_step_from_recon(recon: ReconstructionResult) -> float:
    """Calculate average step size from reconstruction poses.
    
    Returns:
        Average step size in reconstruction units, or 1.0 if insufficient data
    """
    frames = recon.frames
    if len(frames) < 2:
        return 1.0
    
    try:
        translations = np.array([recon.poses[f.image_id].t for f in frames])
        diffs = np.linalg.norm(np.diff(translations, axis=0), axis=1)
        if diffs.size == 0:
            return 1.0
            
        # Filter out non-finite values
        diffs = diffs[np.isfinite(diffs)]
        if diffs.size == 0:
            return 1.0
            
        avg_step = float(np.mean(diffs))
        return avg_step if np.isfinite(avg_step) else 1.0
    except (KeyError, ValueError, RuntimeError) as e:
        logger.warning(f"Error computing reconstruction step: {e}")
        return 1.0


def _height_from_anchors(frames: Iterable[FrameMeta], anchors: Iterable[Anchor]) -> float:
    """Estimate camera height from ground anchors.
    
    Args:
        frames: Frame metadata with altitude information
        anchors: Ground control points with known heights
        
    Returns:
        Estimated camera height above ground in meters
    """
    frames = [f for f in frames if f.alt_ellip is not None]
    if not frames:
        default_height = (constants.H_MIN_M + constants.H_MAX_M) * 0.5
        logger.debug("No frames with altitude data, using default height")
        return default_height

    try:
        cam_alts = [f.alt_ellip for f in frames]
        # Filter non-finite values
        cam_alts = [alt for alt in cam_alts if np.isfinite(alt)]
        if not cam_alts:
            return (constants.H_MIN_M + constants.H_MAX_M) * 0.5
            
        avg_cam_alt = float(np.mean(cam_alts))
        
        base_alts = []
        for anchor in anchors:
            base_alt = anchor.alt_ellip - anchor.height_m
            if np.isfinite(base_alt):
                base_alts.append(base_alt)
                
        if not base_alts:
            logger.debug("No valid anchors, using default height")
            return (constants.H_MIN_M + constants.H_MAX_M) * 0.5
            
        avg_base_alt = float(np.mean(base_alts))
        height = avg_cam_alt - avg_base_alt
        
        if not np.isfinite(height):
            logger.warning(
                f"Non-finite height computed (cam_alt: {avg_cam_alt}, base_alt: {avg_base_alt})"
            )
            return (constants.H_MIN_M + constants.H_MAX_M) * 0.5
            
        return height
    except (ValueError, RuntimeError) as e:
        logger.warning(f"Error computing height from anchors: {e}")
        return (constants.H_MIN_M + constants.H_MAX_M) * 0.5
