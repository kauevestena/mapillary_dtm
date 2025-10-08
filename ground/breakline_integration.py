"""
3D breakline integration for constrained TIN construction.

This module projects 2D curb/edge detections from images to 3D world coordinates,
merges overlapping segments, and prepares breakline constraints for TIN.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

try:
    from .. import constants
    from ..common_core import FrameMeta
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    import constants
    from common_core import FrameMeta

log = logging.getLogger(__name__)


@dataclass
class Breakline3D:
    """3D breakline polyline in world coordinates."""

    seq_id: str
    type: str  # "curb" | "lane_edge" | "median"
    points: List[Tuple[float, float, float]]  # (X, Y, Z) in ENU
    confidence: float
    image_ids: List[str]  # Source images


def project_curbs_to_3d(
    curbs: Mapping[str, List],  # Output from curb_edge_lane.extract_curbs_and_lanes
    camera_poses: Mapping[str, Mapping[str, object]],  # image_id -> pose dict
    camera_models: Mapping[str, Mapping[str, object]],  # image_id -> camera dict
    consensus_points: Sequence[Mapping[str, object]],
    max_height_dev_m: float | None = None,
) -> List[Breakline3D]:
    """Project 2D curb detections to 3D world coordinates.

    For each curb point in image space:
    1. Compute ray from camera center through image point
    2. Intersect ray with local ground plane (estimated from consensus)
    3. Filter outliers based on height deviation from nearby points

    Args:
        curbs: Dictionary of sequence_id -> List[CurbLine]
        camera_poses: Dictionary of image_id -> pose (rotation + translation)
        camera_models: Dictionary of image_id -> camera intrinsics
        consensus_points: Ground points for local height reference
        max_height_dev_m: Maximum height deviation for outlier filtering

    Returns:
        List of Breakline3D objects with 3D positions
    """
    max_dev = max_height_dev_m or constants.BREAKLINE_MAX_HEIGHT_DEV_M

    # Build spatial index for consensus points
    if consensus_points:
        consensus_xy = np.array([[p["x"], p["y"]] for p in consensus_points])
        consensus_z = np.array([p["z"] for p in consensus_points])
        tree = cKDTree(consensus_xy)
    else:
        tree = None

    breaklines: List[Breakline3D] = []

    for seq_id, curb_list in curbs.items():
        for curb in curb_list:
            image_id = curb.image_id

            # Get camera pose and model
            pose = camera_poses.get(image_id)
            camera = camera_models.get(image_id)

            if pose is None or camera is None:
                log.debug("Missing pose/camera for image %s, skipping curb", image_id)
                continue

            # Extract camera parameters
            R = np.array(pose.get("rotation", np.eye(3)))  # 3x3 rotation matrix
            t = np.array(pose.get("translation", [0, 0, 0]))  # 3D translation
            C = -R.T @ t  # Camera center in world coordinates

            # Get image dimensions and focal length
            width = int(camera.get("width", 1920))
            height = int(camera.get("height", 1080))
            focal = float(camera.get("focal", 0.8))  # Normalized focal length

            # Project each curb point
            points_3d: List[Tuple[float, float, float]] = []

            for x_norm, y_norm in curb.xy_norm:
                # Convert normalized [0,1] to pixel coordinates
                x_px = x_norm * width
                y_px = y_norm * height

                # Ray direction in camera coordinates (pinhole model)
                # Assuming principal point at image center
                ray_cam = np.array(
                    [
                        (x_px - width / 2) / (focal * width / 2),
                        (y_px - height / 2) / (focal * width / 2),
                        1.0,
                    ]
                )
                ray_cam = ray_cam / np.linalg.norm(ray_cam)

                # Transform ray to world coordinates
                ray_world = R.T @ ray_cam

                # Estimate ground height at ray position
                if tree is not None and len(consensus_xy) > 0:
                    # Find nearby consensus points
                    dists, indices = tree.query(C[:2], k=min(10, len(consensus_xy)))
                    nearby_z = consensus_z[indices]
                    local_ground_z = float(np.median(nearby_z))
                else:
                    # Fallback: use camera height - 1.5m (typical)
                    local_ground_z = C[2] - 1.5

                # Intersect ray with horizontal plane at local_ground_z
                # C + t * ray = (X, Y, local_ground_z)
                # C[2] + t * ray[2] = local_ground_z
                if abs(ray_world[2]) < 1e-6:
                    # Ray nearly parallel to ground - skip
                    continue

                t_intersect = (local_ground_z - C[2]) / ray_world[2]

                if t_intersect < 0:
                    # Intersection behind camera - skip
                    continue

                # 3D point on ground
                X, Y, Z = C + t_intersect * ray_world

                # Filter outliers: check if Z is reasonable
                if tree is not None and len(consensus_xy) > 0:
                    # Check against nearby consensus points
                    dists_pt, indices_pt = tree.query(
                        [X, Y], k=min(5, len(consensus_xy))
                    )
                    nearby_z_pt = consensus_z[indices_pt]
                    median_z_pt = float(np.median(nearby_z_pt))

                    if abs(Z - median_z_pt) > max_dev:
                        # Outlier - skip
                        continue

                points_3d.append((float(X), float(Y), float(Z)))

            if len(points_3d) < 2:
                # Not enough valid points
                continue

            breaklines.append(
                Breakline3D(
                    seq_id=seq_id,
                    type="curb",  # Default type; can be refined later
                    points=points_3d,
                    confidence=curb.confidence,
                    image_ids=[image_id],
                )
            )

    log.info("Projected %d curb detections to 3D", len(breaklines))
    return breaklines


def merge_breakline_segments(
    breaklines: List[Breakline3D],
    merge_dist_m: float | None = None,
    min_length_m: float | None = None,
) -> List[Breakline3D]:
    """Merge overlapping breakline segments from multiple views.

    Segments are merged if their endpoints are within merge_dist_m.
    Short segments (< min_length_m) are discarded.

    Args:
        breaklines: List of raw Breakline3D objects
        merge_dist_m: Maximum distance to merge segments
        min_length_m: Minimum segment length to keep

    Returns:
        List of merged Breakline3D objects
    """
    merge_dist = merge_dist_m or constants.BREAKLINE_MERGE_DIST_M
    min_length = min_length_m or constants.BREAKLINE_MIN_LENGTH_M

    if not breaklines:
        return []

    # Filter by length first
    filtered = []
    for bl in breaklines:
        length = _polyline_length(bl.points)
        if length >= min_length:
            filtered.append(bl)

    log.info(
        "Filtered %d/%d breaklines by minimum length %.2fm",
        len(filtered),
        len(breaklines),
        min_length,
    )

    # Simple merging strategy: group by proximity
    # More sophisticated: use graph-based clustering
    merged: List[Breakline3D] = []
    used = set()

    for i, bl1 in enumerate(filtered):
        if i in used:
            continue

        # Start a new merged segment
        group = [bl1]
        used.add(i)

        # Find nearby segments
        for j, bl2 in enumerate(filtered):
            if j in used or j <= i:
                continue

            # Check if endpoints are close
            if _segments_overlap(bl1.points, bl2.points, merge_dist):
                group.append(bl2)
                used.add(j)

        # Merge the group
        merged_points = _merge_polylines([bl.points for bl in group])
        merged_conf = np.mean([bl.confidence for bl in group])
        merged_ids = [img_id for bl in group for img_id in bl.image_ids]

        merged.append(
            Breakline3D(
                seq_id=group[0].seq_id,
                type=group[0].type,
                points=merged_points,
                confidence=float(merged_conf),
                image_ids=merged_ids,
            )
        )

    log.info("Merged %d segments into %d polylines", len(filtered), len(merged))
    return merged


def simplify_breaklines(
    breaklines: List[Breakline3D],
    tolerance_m: float | None = None,
) -> List[Breakline3D]:
    """Simplify breakline polylines using Douglas-Peucker algorithm.

    Args:
        breaklines: List of Breakline3D objects
        tolerance_m: Simplification tolerance (perpendicular distance)

    Returns:
        List of simplified Breakline3D objects
    """
    tol = tolerance_m or constants.BREAKLINE_SIMPLIFY_TOL_M

    simplified = []
    for bl in breaklines:
        simple_pts = _douglas_peucker_3d(bl.points, tol)
        if len(simple_pts) >= 2:
            simplified.append(
                Breakline3D(
                    seq_id=bl.seq_id,
                    type=bl.type,
                    points=simple_pts,
                    confidence=bl.confidence,
                    image_ids=bl.image_ids,
                )
            )

    total_before = sum(len(bl.points) for bl in breaklines)
    total_after = sum(len(bl.points) for bl in simplified)
    log.info(
        "Simplified breaklines: %d → %d vertices (%.1f%% reduction)",
        total_before,
        total_after,
        100 * (1 - total_after / max(total_before, 1)),
    )

    return simplified


def densify_breaklines(
    breaklines: List[Breakline3D],
    max_spacing_m: float | None = None,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Densify breaklines to uniform spacing and build edge connectivity.

    Args:
        breaklines: List of Breakline3D objects
        max_spacing_m: Maximum spacing between vertices

    Returns:
        vertices: (N, 3) array of XYZ positions
        edges: List of (i, j) vertex index pairs
    """
    max_spacing = max_spacing_m or constants.BREAKLINE_DENSIFY_MAX_SPACING_M

    all_vertices: List[Tuple[float, float, float]] = []
    all_edges: List[Tuple[int, int]] = []
    vertex_offset = 0

    for bl in breaklines:
        # Resample polyline to uniform spacing
        resampled = _resample_polyline_3d(bl.points, max_spacing)

        # Add vertices
        all_vertices.extend(resampled)

        # Add edges
        for i in range(len(resampled) - 1):
            all_edges.append((vertex_offset + i, vertex_offset + i + 1))

        vertex_offset += len(resampled)

    vertices = np.array(all_vertices, dtype=np.float64)

    log.info(
        "Densified breaklines: %d vertices, %d edges", len(vertices), len(all_edges)
    )
    return vertices, all_edges


# --- Helper functions ---


def _polyline_length(points: List[Tuple[float, float, float]]) -> float:
    """Calculate total length of a 3D polyline."""
    if len(points) < 2:
        return 0.0
    length = 0.0
    for i in range(len(points) - 1):
        dx = points[i + 1][0] - points[i][0]
        dy = points[i + 1][1] - points[i][1]
        dz = points[i + 1][2] - points[i][2]
        length += math.sqrt(dx * dx + dy * dy + dz * dz)
    return length


def _segments_overlap(
    pts1: List[Tuple[float, float, float]],
    pts2: List[Tuple[float, float, float]],
    threshold: float,
) -> bool:
    """Check if two polylines have endpoints within threshold distance."""
    if not pts1 or not pts2:
        return False

    # Check all endpoint pairs (first/last of each segment)
    endpoints1 = [pts1[0], pts1[-1]]
    endpoints2 = [pts2[0], pts2[-1]]

    for p1 in endpoints1:
        for p2 in endpoints2:
            dx = p1[0] - p2[0]
            dy = p1[1] - p2[1]
            dz = p1[2] - p2[2]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist < threshold:
                return True
    return False


def _merge_polylines(
    polylines: List[List[Tuple[float, float, float]]],
) -> List[Tuple[float, float, float]]:
    """Merge multiple polylines into one by averaging nearby points."""
    if len(polylines) == 1:
        return polylines[0]

    # Simple approach: concatenate and average nearby points
    # More sophisticated: use graph-based ordering
    all_points = [pt for poly in polylines for pt in poly]

    if not all_points:
        return []

    # Build spatial index
    xy = np.array([[p[0], p[1]] for p in all_points])
    tree = cKDTree(xy)

    # Cluster nearby points
    merged: List[Tuple[float, float, float]] = []
    used = set()

    for i, pt in enumerate(all_points):
        if i in used:
            continue

        # Find nearby points
        indices = tree.query_ball_point([pt[0], pt[1]], r=0.3)  # 30cm clustering

        # Average positions
        cluster = [all_points[j] for j in indices if j not in used]
        if cluster:
            x_avg = np.mean([p[0] for p in cluster])
            y_avg = np.mean([p[1] for p in cluster])
            z_avg = np.mean([p[2] for p in cluster])
            merged.append((float(x_avg), float(y_avg), float(z_avg)))

            for j in indices:
                used.add(j)

    return merged


def _douglas_peucker_3d(
    points: List[Tuple[float, float, float]], tolerance: float
) -> List[Tuple[float, float, float]]:
    """Douglas-Peucker polyline simplification in 3D."""
    if len(points) < 3:
        return points

    # Find point with maximum perpendicular distance from line
    first = np.array(points[0])
    last = np.array(points[-1])

    max_dist = 0.0
    max_idx = 0

    for i in range(1, len(points) - 1):
        pt = np.array(points[i])
        dist = _point_line_distance_3d(pt, first, last)
        if dist > max_dist:
            max_dist = dist
            max_idx = i

    # If max distance is greater than tolerance, recursively simplify
    if max_dist > tolerance:
        # Recursive call
        left = _douglas_peucker_3d(points[: max_idx + 1], tolerance)
        right = _douglas_peucker_3d(points[max_idx:], tolerance)
        # Concatenate without duplicating the middle point
        return left[:-1] + right
    else:
        # All points are close enough - keep only endpoints
        return [points[0], points[-1]]


def _point_line_distance_3d(
    point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray
) -> float:
    """Calculate perpendicular distance from point to line segment in 3D."""
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-9:
        return np.linalg.norm(point - line_start)

    # Project point onto line
    t = np.dot(point - line_start, line_vec) / (line_len * line_len)
    t = np.clip(t, 0.0, 1.0)  # Clamp to segment

    projection = line_start + t * line_vec
    return float(np.linalg.norm(point - projection))


def _resample_polyline_3d(
    points: List[Tuple[float, float, float]], max_spacing: float
) -> List[Tuple[float, float, float]]:
    """Resample polyline to uniform spacing."""
    if len(points) < 2:
        return points

    resampled = [points[0]]

    for i in range(len(points) - 1):
        p0 = np.array(points[i])
        p1 = np.array(points[i + 1])

        segment_vec = p1 - p0
        segment_len = np.linalg.norm(segment_vec)

        if segment_len < 1e-9:
            continue

        # Number of intermediate points needed
        n_points = int(np.ceil(segment_len / max_spacing))

        for j in range(1, n_points):
            t = j / n_points
            new_pt = p0 + t * segment_vec
            resampled.append((float(new_pt[0]), float(new_pt[1]), float(new_pt[2])))

        # Add endpoint (unless it's a duplicate)
        if i == len(points) - 2:
            resampled.append(points[-1])

    return resampled
