"""Adapter for invoking OpenSfM or loading canned reconstruction fixtures."""

from __future__ import annotations

import json
import logging
import math
import os
import shlex
import shutil
import signal
import subprocess
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
        self.docker_image = os.getenv("OPEN_SFM_DOCKER_IMAGE", "freakthemighty/opensfm:latest")
        base_root = (
            Path(workspace_root)
            if workspace_root is not None
            else Path(constants.MAPILLARY_CACHE_ROOT) / "opensfm"
        )
        self.workspace_root = base_root.resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Availability checks
    def is_available(self) -> bool:
        return shutil.which(self.opensfm_cmd) is not None or shutil.which("docker") is not None

    # ------------------------------------------------------------------
    # Entry point
    def reconstruct(
        self,
        sequences: Mapping[str, Iterable[FrameMeta]],
        *,
        fixture_path: Optional[Path | str] = None,
        force: bool = False,
        imagery_root: Optional[Path | str] = None,
        progress: bool = False,
    ) -> Dict[str, ReconstructionResult]:
        """Run OpenSfM or load a fixture to obtain reconstruction results."""
        if fixture_path:
            return self._load_fixture(Path(fixture_path), sequences)

        if not self.is_available():
            raise OpenSfMUnavailable(
                "OpenSfM binary not found on PATH; set OPEN_SFM_BIN or provide fixture_path."
            )

        return self._run_binary(sequences, force=force, imagery_root=imagery_root, progress=progress)

    # ------------------------------------------------------------------
    def _run_binary(
        self,
        sequences: Mapping[str, Iterable[FrameMeta]],
        *,
        force: bool,
        imagery_root: Optional[Path | str],
        progress: bool,
    ) -> Dict[str, ReconstructionResult]:
        """Stage imagery, run OpenSfM, and parse reconstruction outputs."""
        results: Dict[str, ReconstructionResult] = {}
        timeout = int(os.getenv("OPEN_SFM_TIMEOUT_SEC", "7200"))
        for seq_id, frames_iter in _progress_iter(sequences.items(), "OpenSfM", progress):
            frames = list(frames_iter)
            if not frames:
                continue
            dataset_dir = self._prepare_dataset(seq_id, frames, imagery_root=imagery_root, force=force)
            reconstruction_path: Path | None = None
            seq_result: Dict[str, ReconstructionResult] = {}
            if not force:
                try:
                    reconstruction_path = self._find_reconstruction(dataset_dir)
                    log.info("Reusing existing OpenSfM reconstruction at %s", reconstruction_path)
                    seq_result = self._load_fixture(reconstruction_path, {seq_id: frames})
                except Exception:
                    reconstruction_path = None
                    seq_result = {}
            if reconstruction_path is None or seq_id not in seq_result:
                if reconstruction_path is not None:
                    log.info("Cached OpenSfM reconstruction for %s is invalid or empty; re-running.", seq_id)
                try:
                    self._run_workflow(dataset_dir, frames, timeout=timeout)
                    reconstruction_path = self._find_reconstruction(dataset_dir)
                    seq_result = self._load_fixture(reconstruction_path, {seq_id: frames})
                except subprocess.CalledProcessError as exc:
                    log.warning("OpenSfM failed for %s: %s", seq_id, exc)
                    continue
                except subprocess.TimeoutExpired as exc:
                    log.warning("OpenSfM timed out for %s: %s", seq_id, exc)
                    continue
                except OpenSfMUnavailable as exc:
                    log.warning("OpenSfM unavailable or produced no reconstruction for %s: %s", seq_id, exc)
                    continue

            for result in seq_result.values():
                result.metadata["source_type"] = "real"
                result.metadata["workspace"] = str(dataset_dir)
            results.update(seq_result)
        return results

    def _prepare_dataset(
        self,
        seq_id: str,
        frames: list[FrameMeta],
        *,
        imagery_root: Optional[Path | str],
        force: bool,
    ) -> Path:
        dataset_dir = self.workspace_root / str(seq_id)
        images_dir = dataset_dir / "images"
        if force and dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        images_dir.mkdir(parents=True, exist_ok=True)
        using_docker = self._using_docker()

        staged = 0
        for frame in frames:
            src = _find_cached_image(frame, imagery_root)
            if src is None:
                continue
            dest = images_dir / f"{frame.image_id}{src.suffix.lower()}"
            if using_docker and dest.is_symlink():
                dest.unlink()
            if not dest.exists():
                if using_docker:
                    shutil.copy2(src, dest)
                    staged += 1
                    continue
                try:
                    os.symlink(src.resolve(), dest)
                except OSError:
                    shutil.copy2(src, dest)
            staged += 1

        if staged < 2:
            raise OpenSfMUnavailable(
                f"OpenSfM needs at least two staged images for {seq_id}; staged {staged}."
            )
        _write_config(dataset_dir)
        if not (dataset_dir / "reconstruction.json").exists():
            _clear_interrupted_outputs(dataset_dir)
        return dataset_dir

    def _using_docker(self) -> bool:
        return shutil.which(self.opensfm_cmd) is None and shutil.which("docker") is not None

    def _run_workflow(self, dataset_dir: Path, frames: list[FrameMeta], *, timeout: int) -> None:
        """Run OpenSfM commands, injecting trusted frame metadata before matching."""
        if shutil.which(self.opensfm_cmd):
            cmd = [self.opensfm_cmd, str(dataset_dir)]
            _run_logged(cmd, cwd=dataset_dir, timeout=timeout, log_path=dataset_dir / "opensfm_command.log")
            return
        if not shutil.which("docker"):
            raise OpenSfMUnavailable(
                "Neither opensfm_run_all nor docker is available for OpenSfM execution."
            )

        log_path = dataset_dir / "opensfm_command.log"
        self._repair_docker_permissions(dataset_dir, log_path=log_path)
        for command_name in ("extract_metadata", "detect_features", "match_features", "create_tracks", "reconstruct"):
            cmd = self._docker_command(dataset_dir, command_name)
            _run_logged(cmd, cwd=dataset_dir, timeout=timeout, log_path=log_path, docker_cidfile=dataset_dir / "opensfm_container.cid")
            if command_name == "extract_metadata":
                self._repair_docker_permissions(dataset_dir, log_path=log_path)
                _patch_exif_metadata(dataset_dir, frames)
        if _env_truthy("OPEN_SFM_RUN_MESH"):
            _run_logged(
                self._docker_command(dataset_dir, "mesh"),
                cwd=dataset_dir,
                timeout=timeout,
                log_path=log_path,
                docker_cidfile=dataset_dir / "opensfm_container.cid",
            )

    def _docker_command(self, dataset_dir: Path, command_name: str) -> list[str]:
        cidfile = dataset_dir / "opensfm_container.cid"
        cidfile.unlink(missing_ok=True)
        return [
            "docker",
            "run",
            "--rm",
            "--cidfile",
            str(cidfile),
            "-v",
            f"{dataset_dir}:/data",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-e",
            "HOME=/tmp",
            self.docker_image,
            *_opensfm_docker_command(self.docker_image, command_name),
            "/data",
        ]

    def _repair_docker_permissions(self, dataset_dir: Path, *, log_path: Path) -> None:
        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{dataset_dir}:/data",
            self.docker_image,
            "chown",
            "-R",
            f"{os.getuid()}:{os.getgid()}",
            "/data",
        ]
        _run_logged(cmd, cwd=dataset_dir, timeout=300, log_path=log_path)

    @staticmethod
    def _find_reconstruction(dataset_dir: Path) -> Path:
        candidates = [
            dataset_dir / "reconstruction.json",
            dataset_dir / "reconstruction" / "reconstruction.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        matches = sorted(dataset_dir.rglob("reconstruction.json"))
        if matches:
            return matches[0]
        raise OpenSfMUnavailable(f"OpenSfM produced no reconstruction.json under {dataset_dir}")

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

        for image_filename, shot in shots.items():
            image_id = image_filename.rsplit('.', 1)[0] if '.' in image_filename else image_filename
            frame = frames_by_id.get(image_id)
            if frame is None:
                log.debug("Fixture shot %s not found in sequences; skipping", image_filename)
                continue

            seq_id = frame.seq_id
            rotation = shot.get("rotation") or shot.get("quaternion") or [1.0, 0.0, 0.0, 0.0]
            translation = shot.get("translation") or [0.0, 0.0, 0.0]
            if len(translation) != 3:
                raise ValueError(f"Shot {image_id} translation must provide 3 values")
            if len(rotation) == 4:
                R = _quat_to_matrix(rotation)
                pose_t = np.asarray(translation, dtype=np.float64)
            elif len(rotation) == 3:
                R_cw = _rodrigues_to_matrix(rotation)
                R = R_cw.T
                pose_t = -R @ np.asarray(translation, dtype=np.float64)
            else:
                raise ValueError(f"Shot {image_id} rotation must provide 3 or 4 values")
            pose = Pose(R=R, t=pose_t)

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
                metadata={"fixture": str(fixture_path), "coordinate_frame": "enu", "source_type": "fixture"},
            )

        return results


def _rodrigues_to_matrix(rotvec: Iterable[float]) -> np.ndarray:
    vec = np.asarray(list(rotvec), dtype=np.float64)
    theta = float(np.linalg.norm(vec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = vec / theta
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def _find_cached_image(frame: FrameMeta, imagery_root: Optional[Path | str]) -> Optional[Path]:
    from ..ingest.image_loader import ImageryLoader
    loader = ImageryLoader(imagery_root)
    seq_dir = loader._sequence_dir(frame.seq_id)
    for path in loader._candidate_paths(seq_dir, frame.image_id):
        if path.exists():
            return path
    return None


def _write_config(dataset_dir: Path) -> None:
    config = {
        "matching_gps_distance": _env_float("OPEN_SFM_MATCHING_GPS_DISTANCE_M", 80.0),
        "matching_gps_neighbors": _env_int("OPEN_SFM_MATCHING_GPS_NEIGHBORS", 8),
        "matching_time_neighbors": _env_int("OPEN_SFM_MATCHING_TIME_NEIGHBORS", 8),
        "processes": _env_int("OPEN_SFM_PROCESSES", min(4, max(1, os.cpu_count() or 1))),
        "feature_process_size": _env_int("OPEN_SFM_FEATURE_PROCESS_SIZE", 2048),
        "bundle_interval": _env_int("OPEN_SFM_BUNDLE_INTERVAL", 10),
        "bundle_new_points_ratio": _env_float("OPEN_SFM_BUNDLE_NEW_POINTS_RATIO", 2.0),
        "use_altitude_tag": "yes",
    }
    lines = [
        "# Generated by dtm_from_mapillary for reproducible production resumes.",
        "# Override with OPEN_SFM_MATCHING_* and OPEN_SFM_PROCESSES environment variables.",
    ]
    for key, value in config.items():
        lines.append(f"{key}: {value}")
    (dataset_dir / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf8")
    (dataset_dir / "opensfm_config_provenance.json").write_text(
        json.dumps({"source": "dtm_from_mapillary", "config": config}, indent=2),
        encoding="utf8",
    )


def _clear_interrupted_outputs(dataset_dir: Path) -> None:
    for name in (
        "matches",
        "tracks.csv",
        "tracks.csv.gz",
        "reconstruction",
        "reconstruction.json",
        "undistorted",
        "depthmaps",
    ):
        path = dataset_dir / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def _patch_exif_metadata(dataset_dir: Path, frames: list[FrameMeta]) -> None:
    patched = 0
    missing = []
    for frame in frames:
        matches = sorted((dataset_dir / "exif").glob(f"{frame.image_id}.*.exif"))
        if not matches:
            missing.append(frame.image_id)
            continue
        path = matches[0]
        payload = json.loads(path.read_text(encoding="utf8"))
        payload["capture_time"] = float(frame.captured_at_ms) / 1000.0
        gps = {
            "latitude": float(frame.lat),
            "longitude": float(frame.lon),
            "dop": float(frame.cam_params.get("gps_dop", 5.0)),
        }
        if frame.alt_ellip is not None:
            gps["altitude"] = float(frame.alt_ellip)
        payload["gps"] = gps
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf8")
        patched += 1
    provenance = {
        "source": "FrameMeta",
        "patched": patched,
        "missing": missing[:20],
        "missing_count": len(missing),
    }
    (dataset_dir / "exif_metadata_overrides.json").write_text(
        json.dumps(provenance, indent=2),
        encoding="utf8",
    )
    if missing:
        raise OpenSfMUnavailable(
            f"OpenSfM metadata patch missing EXIF files for {len(missing)} staged frames"
        )


def _progress_iter(items, desc: str, enabled: bool):
    if not enabled:
        return items
    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover - optional display dependency
        return items
    return tqdm(items, desc=desc, unit="seq")


def _opensfm_docker_command(image: str, command_name: str) -> list[str]:
    raw = os.getenv("OPEN_SFM_DOCKER_CMD")
    if raw:
        text = raw.replace("{command}", command_name)
        parts = shlex.split(text)
        if "{command}" not in raw:
            parts.append(command_name)
        return parts
    if "freakthemighty/opensfm" in image:
        return ["/source/OpenSfM/bin/opensfm", command_name]
    return ["opensfm", command_name]


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise OpenSfMUnavailable(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise OpenSfMUnavailable(f"{name} must be a float, got {raw!r}") from exc


def _run_logged(
    cmd: list[str],
    *,
    cwd: Path,
    timeout: int,
    log_path: Path,
    docker_cidfile: Path | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as stream:
        stream.write(("\n$ " + " ".join(cmd) + "\n").encode("utf8", errors="replace"))
        stream.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return_code = proc.wait(timeout=timeout)
        except (BaseException, subprocess.TimeoutExpired):
            _terminate_process(proc, docker_cidfile)
            raise
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)
        if docker_cidfile is not None:
            docker_cidfile.unlink(missing_ok=True)


def _terminate_process(proc: subprocess.Popen, docker_cidfile: Path | None) -> None:
    if docker_cidfile is not None and docker_cidfile.exists():
        cid = docker_cidfile.read_text(encoding="utf8", errors="replace").strip()
        if cid:
            subprocess.run(
                ["docker", "stop", cid],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
                check=False,
            )
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=10)
    if docker_cidfile is not None:
        docker_cidfile.unlink(missing_ok=True)
