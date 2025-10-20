"""Filter Mapillary sequences to fast-moving (likely car) segments."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, NamedTuple

try:
    from pyproj import Geod
except ImportError as exc:  # pragma: no cover - handled at runtime
    Geod = None
    _GEOD = None
    _GEOD_IMPORT_ERROR = exc
else:
    _GEOD = Geod(ellps="WGS84")
    _GEOD_IMPORT_ERROR = None

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

from .. import constants
from ..common_core import FrameMeta

log = logging.getLogger(__name__)


class SpeedStatistics(NamedTuple):
    """Detailed speed statistics for a sequence."""

    min_kmh: float
    q1_kmh: float
    median_kmh: float
    q3_kmh: float
    max_kmh: float
    mean_kmh: float
    std_kmh: float
    sample_count: int


def compute_speed_statistics(speeds: List[float]) -> Optional[SpeedStatistics]:
    """Compute comprehensive speed statistics from a list of speeds.

    Args:
        speeds: List of speeds in km/h

    Returns:
        SpeedStatistics namedtuple with min, q1, median, q3, max, mean, std, count
        Returns None if no valid speeds provided
    """
    if not speeds:
        return None

    if np is not None:
        # Use numpy for accurate percentiles
        speeds_arr = np.array(speeds)
        return SpeedStatistics(
            min_kmh=float(np.min(speeds_arr)),
            q1_kmh=float(np.percentile(speeds_arr, 25)),
            median_kmh=float(np.median(speeds_arr)),
            q3_kmh=float(np.percentile(speeds_arr, 75)),
            max_kmh=float(np.max(speeds_arr)),
            mean_kmh=float(np.mean(speeds_arr)),
            std_kmh=float(np.std(speeds_arr)),
            sample_count=len(speeds),
        )
    else:
        # Fallback to pure Python (less accurate for percentiles)
        sorted_speeds = sorted(speeds)
        n = len(sorted_speeds)

        def percentile(data: List[float], p: float) -> float:
            """Simple percentile calculation."""
            k = (n - 1) * p
            f = int(k)
            c = f + 1 if f + 1 < n else f
            if f == c:
                return data[f]
            return data[f] + (k - f) * (data[c] - data[f])

        mean = sum(speeds) / n
        variance = sum((x - mean) ** 2 for x in speeds) / n
        std = variance**0.5

        return SpeedStatistics(
            min_kmh=sorted_speeds[0],
            q1_kmh=percentile(sorted_speeds, 0.25),
            median_kmh=percentile(sorted_speeds, 0.50),
            q3_kmh=percentile(sorted_speeds, 0.75),
            max_kmh=sorted_speeds[-1],
            mean_kmh=mean,
            std_kmh=std,
            sample_count=n,
        )


def filter_car_sequences(
    seqs: Dict[str, List[FrameMeta]],
    min_speed_kmh: float | None = None,
    max_speed_kmh: float | None = None,
    return_statistics: bool = False,
) -> (
    Dict[str, List[FrameMeta]]
    | tuple[Dict[str, List[FrameMeta]], Dict[str, SpeedStatistics]]
):
    """Filter sequences by max speed achieved, camera type, and quality score.

    The original design: compute all frame-to-frame speeds in a sequence,
    then judge the maximum speed achieved. If max_speed ∈ [min_speed, max_speed],
    the entire sequence is kept (after per-frame quality/camera filtering).

    Args:
        seqs: Dictionary mapping sequence IDs to lists of FrameMeta
        min_speed_kmh: Minimum max-speed threshold (default: constants.MIN_SPEED_KMH)
        max_speed_kmh: Maximum max-speed threshold (default: constants.MAX_SPEED_KMH)
        return_statistics: If True, also return detailed speed statistics per sequence

    Returns:
        If return_statistics is False: Filtered dictionary with car-only sequences
        If return_statistics is True: Tuple of (filtered_sequences, speed_statistics)
    """
    if _GEOD is None:
        raise RuntimeError(
            "pyproj is required for filter_car_sequences"
        ) from _GEOD_IMPORT_ERROR

    # Use provided thresholds or fall back to constants
    min_speed = min_speed_kmh if min_speed_kmh is not None else constants.MIN_SPEED_KMH
    max_speed = max_speed_kmh if max_speed_kmh is not None else constants.MAX_SPEED_KMH

    filtered: Dict[str, List[FrameMeta]] = {}
    statistics: Dict[str, SpeedStatistics] = {}

    for seq_id, frames in seqs.items():
        if not frames:
            continue

        ordered = sorted(frames, key=lambda f: f.captured_at_ms)

        # Compute all frame-to-frame speeds in the sequence
        all_speeds: List[float] = []
        for idx in range(len(ordered) - 1):
            f0, f1 = ordered[idx], ordered[idx + 1]
            dt_s = max((f1.captured_at_ms - f0.captured_at_ms) / 1000.0, 0.0)
            if dt_s <= 0.1:  # ignore stalled or duplicate timestamps
                continue
            try:
                _, _, dist_m = _GEOD.inv(f0.lon, f0.lat, f1.lon, f1.lat)
            except Exception:  # pragma: no cover - pyproj raises rarely
                continue
            speed_kmh = (dist_m / dt_s) * 3.6
            all_speeds.append(speed_kmh)

        # Compute comprehensive statistics for this sequence
        if all_speeds:
            seq_stats = compute_speed_statistics(all_speeds)
            if seq_stats:
                statistics[seq_id] = seq_stats

        # Judge the sequence by its maximum speed achieved
        if not all_speeds:
            continue

        max_speed_achieved = max(all_speeds)
        if max_speed_achieved < min_speed or max_speed_achieved > max_speed:
            log.debug(
                "Sequence %s rejected: max_speed=%.1f km/h (threshold: %.1f-%.1f km/h)",
                seq_id,
                max_speed_achieved,
                min_speed,
                max_speed,
            )
            continue

        # Sequence passes speed test - now filter individual frames by quality/camera
        kept_frames: List[FrameMeta] = []
        for frame in ordered:
            if (
                frame.camera_type
                and frame.camera_type.lower() not in constants.ALLOW_CAMERA_TYPES
            ):
                continue
            if (
                frame.quality_score is not None
                and frame.quality_score < constants.QUALITY_SCORE_MIN
            ):
                continue
            kept_frames.append(frame)

        if kept_frames:
            filtered[seq_id] = kept_frames
            log.debug(
                "Sequence %s kept: max_speed=%.1f km/h, %d/%d frames after quality filter",
                seq_id,
                max_speed_achieved,
                len(kept_frames),
                len(ordered),
            )

    log.info(
        "Car filter reduced %d sequences to %d (speed range: %.1f-%.1f km/h)",
        len(seqs),
        len(filtered),
        min_speed,
        max_speed,
    )

    if return_statistics:
        return filtered, statistics
    return filtered
