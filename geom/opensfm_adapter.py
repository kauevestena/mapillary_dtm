"""Adapter for invoking OpenSfM or loading canned reconstruction fixtures."""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import numpy as np

from .. import constants
from ..common_core import FrameMeta, Pose, ReconstructionResult

log = logging.getLogger(__name__)


class OpenSfMUnavailable(RuntimeError):
    """Raised when OpenSfM binaries or environment are not available."""


def _quat_to_matrix(q: Iterable[float]) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to rotation matrix."""
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class OpenSfMRunner:
    """Lightweight harness to run OpenSfM or load precomputed fixtures."""

    def __init__(
        self,
        opensfm_cmd: Optional[str] = None,
        workspace_root: Optional[Path | str] = None,
    ) -> None:
        self.opensfm_cmd = opensfm_cmd or os.getenv("OPEN_SFM_BIN") or "opensfm_run_all"
        base_root = (
            Path(workspace_root)
            if workspace_root is not None
            else Path(constants.MAPILLARY_CACHE_ROOT) / "opensfm"
        )
        self.workspace_root = base_root
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Availability checks
    def is_available(self) -> bool:
        return shutil.which(self.opensfm_cmd) is not None

    # ------------------------------------------------------------------
    # Entry point
    def reconstruct(
        self,
        sequences: Mapping[str, Iterable[FrameMeta]],
        *,
        fixture_path: Optional[Path | str] = None,
        force: bool = False,
    ) -> Dict[str, ReconstructionResult]:
        """Run OpenSfM or load a fixture to obtain reconstruction results."""
        if fixture_path:
            return self._load_fixture(Path(fixture_path), sequences)

        if not self.is_available():
            raise OpenSfMUnavailable(
                "OpenSfM binary not found on PATH; set OPEN_SFM_BIN or provide fixture_path."
            )

        return self._run_binary(sequences, force=force)

    # ------------------------------------------------------------------
    def _run_binary(
        self,
        sequences: Mapping[str, Iterable[FrameMeta]],
        *,
        force: bool,
    ) -> Dict[str, ReconstructionResult]:
        """Placeholder for real OpenSfM invocation."""
        raise OpenSfMUnavailable(
            "Real OpenSfM execution not yet wired. Provide OPEN_SFM_FIXTURE or integrate container workflow."
        )

    # ------------------------------------------------------------------
    def _load_fixture(
        self,
        fixture_path: Path,
        sequences: Mapping[str, Iterable[FrameMeta]],
    ) -> Dict[str, ReconstructionResult]:
        if not fixture_path.exists():
            raise FileNotFoundError(f"OpenSfM fixture not found: {fixture_path}")

        payload = json.loads(fixture_path.read_text(encoding="utf8"))
        if not isinstance(payload, list) or not payload:
            raise ValueError("Invalid OpenSfM fixture: expected non-empty list")

        reconstruction = payload[0]
        shots = reconstruction.get("shots") or {}
        cameras = reconstruction.get("cameras") or {}
        points_block = reconstruction.get("points") or {}

        frames_by_id: Dict[str, FrameMeta] = {}
        for seq_id, frames in sequences.items():
            for frame in frames:
                frames_by_id[frame.image_id] = frame

        results: Dict[str, ReconstructionResult] = {}
        per_sequence_frames: Dict[str, list[FrameMeta]] = {}
        per_sequence_poses: Dict[str, Dict[str, Pose]] = {}

        for image_id, shot in shots.items():
            frame = frames_by_id.get(image_id)
            if frame is None:
                log.debug("Fixture shot %s not found in sequences; skipping", image_id)
                continue

            seq_id = frame.seq_id
            rotation = shot.get("rotation") or shot.get("quaternion") or [1.0, 0.0, 0.0, 0.0]
            if len(rotation) != 4:
                raise ValueError(f"Shot {image_id} rotation must provide 4 values")
            R = _quat_to_matrix(rotation)
            translation = shot.get("translation") or [0.0, 0.0, 0.0]
            if len(translation) != 3:
                raise ValueError(f"Shot {image_id} translation must provide 3 values")
            pose = Pose(R=R, t=np.asarray(translation, dtype=np.float64))

            camera_ref = shot.get("camera")
            if camera_ref and camera_ref in cameras:
                cam_info = cameras[camera_ref]
                updated_params = frame.cam_params.copy()
                for key, value in cam_info.items():
                    if key in {"projection_type", "focal", "k1", "k2", "k3", "p1", "p2"} or key.startswith("principal"):
                        updated_params[key] = value
                frame = FrameMeta(
                    image_id=frame.image_id,
                    seq_id=frame.seq_id,
                    captured_at_ms=frame.captured_at_ms,
                    lon=frame.lon,
                    lat=frame.lat,
                    alt_ellip=frame.alt_ellip,
                    camera_type=cam_info.get("projection_type", frame.camera_type),
                    cam_params=updated_params,
                    quality_score=frame.quality_score,
                    thumbnail_url=frame.thumbnail_url,
                )

            per_sequence_frames.setdefault(seq_id, []).append(frame)
            pose_map = per_sequence_poses.setdefault(seq_id, {})
            pose_map[image_id] = pose

        points_xyz = []
        for point in points_block.values():
            coordinates = point.get("coordinates")
            if isinstance(coordinates, (list, tuple)) and len(coordinates) == 3:
                points_xyz.append([float(coordinates[0]), float(coordinates[1]), float(coordinates[2])])

        points_array = (
            np.asarray(points_xyz, dtype=np.float32) if points_xyz else np.zeros((0, 3), dtype=np.float32)
        )

        for seq_id, frames in per_sequence_frames.items():
            frames_sorted = sorted(frames, key=lambda f: f.captured_at_ms)
            poses = per_sequence_poses.get(seq_id, {})
            results[seq_id] = ReconstructionResult(
                seq_id=seq_id,
                frames=frames_sorted,
                poses=poses,
                points_xyz=points_array,
                source="opensfm",
                metadata={"fixture": str(fixture_path), "coordinate_frame": "enu"},
            )

        return results
