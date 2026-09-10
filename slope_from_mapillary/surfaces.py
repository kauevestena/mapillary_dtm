"""Surface labels come from the observed feature pixels, not mask averages."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .reconstruction import Reconstruction, triangulation_angle


@dataclass(frozen=True)
class ObservationPolicy:
    min_views: int = 3
    min_probability: float = 0.8
    min_vote_fraction: float = 0.8
    max_reprojection_error_px: float = 2.0
    min_triangulation_angle_deg: float = 2.0

    def __post_init__(self):
        if not isinstance(self.min_views, int) or self.min_views < 2:
            raise ValueError("min_views must be an integer >= 2")
        if not 0 < self.min_probability <= 1 or not 0.5 < self.min_vote_fraction <= 1:
            raise ValueError("Invalid surface probability/vote thresholds")
        if not np.isfinite(self.max_reprojection_error_px) or self.max_reprojection_error_px <= 0:
            raise ValueError("Reprojection threshold must be finite and positive")
        if not 0 < self.min_triangulation_angle_deg < 90:
            raise ValueError("Triangulation threshold must be in (0, 90)")


@dataclass
class SurfaceCloud:
    xyz: np.ndarray
    point_ids: list[str]
    surface_ids: list[str]
    view_counts: np.ndarray
    stats: dict


def select_surfaces(reconstruction: Reconstruction, mask_dir: Path,
                    surfaces: dict, policy: ObservationPolicy) -> SurfaceCloud:
    """Vote only across track observations. Require class agreement.

    NPZ masks contain `labels`, `confidence`, `image_name`, and `source`.
    Labels are positive project-defined integers; zero is unknown/non-ground.
    A surface ID must identify one physical level, e.g. sidewalk-west-deck-0.
    """
    label_ids = {int(code): str(info["id"]) for code, info in surfaces.items()}
    if not label_ids or min(label_ids) < 1 or len(set(label_ids.values())) != len(label_ids):
        raise ValueError("Surfaces require unique IDs and positive integer label codes")
    masks = {}
    for name, shot in reconstruction.shots.items():
        # basename avoids reading paths outside the selected mask directory.
        path = Path(mask_dir) / (Path(name).name + ".npz")
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as mask:
            if not {"labels", "confidence", "image_name", "source"} <= set(mask.files):
                raise ValueError(f"Mask needs class labels, confidence and provenance: {path}")
            if str(mask["image_name"].item()) != name or not str(mask["source"].item()).strip():
                raise ValueError(f"Mask image/provenance mismatch: {path}")
            labels, confidence = mask["labels"], mask["confidence"]
            if (labels.ndim != 2 or confidence.shape != labels.shape or
                    not np.issubdtype(labels.dtype, np.integer) or min(labels.shape) < 1):
                raise ValueError(f"Invalid mask arrays: {path}")
            if (not np.isfinite(confidence).all() or confidence.min() < 0 or confidence.max() > 1):
                raise ValueError(f"Mask confidence must lie in [0, 1]: {path}")
            if not np.isclose(labels.shape[1] / labels.shape[0], shot.width / shot.height, rtol=0.01):
                raise ValueError(f"Mask was cropped or rotated relative to the SfM image: {path}")
            masks[name] = (labels, confidence)
    xyz, ids, surface_ids, counts = [], [], [], []
    rejected = Counter()
    for point in reconstruction.points.values():
        if len(point.observations) < policy.min_views:
            rejected["insufficient_track_views"] += 1
            continue
        if not np.isfinite(point.error_px) or point.error_px > policy.max_reprojection_error_px:
            rejected["reprojection_error"] += 1
            continue
        if triangulation_angle(point, reconstruction.shots) < policy.min_triangulation_angle_deg:
            rejected["weak_parallax"] += 1
            continue
        votes = Counter()
        for name, (x, y) in point.observations.items():
            if name not in masks:
                continue
            labels, confidence = masks[name]
            shot = reconstruction.shots[name]
            if not (np.isfinite(x) and np.isfinite(y) and 0 <= x < shot.width and 0 <= y < shot.height):
                continue
            col = min(int(x / shot.width * labels.shape[1]), labels.shape[1] - 1)
            row = min(int(y / shot.height * labels.shape[0]), labels.shape[0] - 1)
            label = int(labels[row, col])
            if label in label_ids and confidence[row, col] >= policy.min_probability:
                votes[label] += 1
        if not votes:
            rejected["no_observed_surface_label"] += 1
            continue
        label, count = votes.most_common(1)[0]
        # Missing masks and low-confidence observations are not silently removed
        # from the denominator, which would artificially increase confidence.
        if count < policy.min_views or count / len(point.observations) < policy.min_vote_fraction:
            rejected["insufficient_semantic_agreement"] += 1
            continue
        xyz.append(point.xyz)
        ids.append(point.id)
        surface_ids.append(label_ids[label])
        counts.append(count)
    return SurfaceCloud(np.asarray(xyz, dtype=float).reshape(-1, 3), ids, surface_ids,
                        np.asarray(counts, dtype=int),
                        {"input_points": len(reconstruction.points), "accepted_points": len(ids),
                         "available_masks": len(masks), "rejected": dict(rejected)})
