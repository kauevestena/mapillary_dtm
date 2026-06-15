"""Adapter for invoking COLMAP or loading precomputed sparse models."""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import numpy as np

from .. import constants
from ..common_core import FrameMeta, Pose, ReconstructionResult

log = logging.getLogger(__name__)


class COLMAPUnavailable(RuntimeError):
    """Raised when COLMAP binaries or workspace prerequisites are missing."""


@dataclass(frozen=True)
class COLMAPConfig:
    """Runtime configuration for COLMAP invocations."""

    threads: int = 8
    use_gpu: bool = False
    gpu_index: str = "0"
    matcher: str = "sequential"  # exhaustive|sequential|vocabtree
    workspace_root: Optional[Path | str] = None

    @staticmethod
    def from_kwargs(
        *,
        threads: Optional[int] = None,
        use_gpu: Optional[bool] = None,
        gpu_index: Optional[str] = None,
        matcher: Optional[str] = None,
        workspace_root: Optional[Path | str] = None,
    ) -> "COLMAPConfig":
        return COLMAPConfig(
            threads=threads if threads is not None else COLMAPConfig().threads,
            use_gpu=use_gpu if use_gpu is not None else COLMAPConfig().use_gpu,
            gpu_index=gpu_index if gpu_index is not None else COLMAPConfig().gpu_index,
            matcher=matcher if matcher is not None else COLMAPConfig().matcher,
            workspace_root=workspace_root,
        )


@dataclass
class _ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: list[float]


@dataclass
class _ColmapImage:
    image_id: int
    camera_id: int
    name: str
    rotation_wc: np.ndarray  # world-from-camera rotation
    center: np.ndarray  # camera center in world coordinates


def _quat_to_matrix(q: Iterable[float]) -> np.ndarray:
    """Convert quaternion [qw, qx, qy, qz] to a rotation matrix."""
    qw, qx, qy, qz = q
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n == 0:
        return np.eye(3, dtype=np.float64)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


class COLMAPRunner:
    """Utility to drive COLMAP reconstructions or load fixtures for testing."""

    def __init__(
        self,
        colmap_cmd: Optional[str] = None,
        *,
        config: Optional[COLMAPConfig] = None,
        workspace_root: Optional[Path | str] = None,
    ) -> None:
        self.colmap_cmd = colmap_cmd or os.getenv("COLMAP_BIN") or "colmap"
        cfg = config or COLMAPConfig()
        if workspace_root is not None:
            cfg = COLMAPConfig(
                threads=cfg.threads,
                use_gpu=cfg.use_gpu,
                gpu_index=cfg.gpu_index,
                matcher=cfg.matcher,
                workspace_root=workspace_root,
            )
        self.config = cfg
        base_root = (
            Path(self.config.workspace_root)
            if self.config.workspace_root is not None
            else Path(constants.MAPILLARY_CACHE_ROOT) / "colmap"
        )
        self.workspace_root = base_root.resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        """Check whether the COLMAP binary is discoverable."""
        if shutil.which(self.colmap_cmd) is not None:
            return True
        return self._docker_image_available()

    # ------------------------------------------------------------------
    def reconstruct(
        self,
        sequences: Mapping[str, Iterable[FrameMeta]],
        *,
        fixture_path: Optional[Path | str] = None,
        force: bool = False,
        imagery_root: Optional[Path | str] = None,
        progress: bool = False,
    ) -> Dict[str, ReconstructionResult]:
        """Run COLMAP or load a canned sparse model."""
        if fixture_path:
            results = self._load_fixture(Path(fixture_path), sequences)
            for result in results.values():
                result.metadata["source_type"] = "fixture"
            return results

        if not self.is_available():
            raise COLMAPUnavailable(
                "COLMAP binary not found on PATH and COLMAP_DOCKER_IMAGE is not available; "
                "set COLMAP_BIN, pull/set COLMAP_DOCKER_IMAGE, or provide fixture_path."
            )

        workspace = self._prepare_workspace(force=force)
        sparse_dir = workspace / "sparse" / "0"
        model_exists = sparse_dir.exists() and any(sparse_dir.glob("*.txt"))

        if model_exists and not force:
            log.info("Reusing existing COLMAP sparse model at %s", sparse_dir)
        else:
            self._run_binary_pipeline(workspace, sequences, imagery_root=imagery_root, progress=progress)

        txt_dir = workspace / "sparse_txt" / "0"
        if not any(sparse_dir.glob("*.txt")):
            self._convert_model_to_text(sparse_dir, txt_dir)
            sparse_dir = txt_dir
        results = self._load_fixture(sparse_dir, sequences)
        for result in results.values():
            result.metadata["source_type"] = "real"
            result.metadata["workspace"] = str(workspace)
        return results

    # ------------------------------------------------------------------
    def _prepare_workspace(self, *, force: bool) -> Path:
        """Create or reuse a workspace layout for COLMAP runs."""
        workspace = self.workspace_root / "workspace"
        (workspace / "images").mkdir(parents=True, exist_ok=True)
        (workspace / "sparse").mkdir(parents=True, exist_ok=True)
        database = workspace / "database.db"
        if force and database.exists():
            database.unlink()
        return workspace

    # ------------------------------------------------------------------
    def _run_binary_pipeline(
        self,
        workspace: Path,
        sequences: Mapping[str, Iterable[FrameMeta]],
        *,
        imagery_root: Optional[Path | str],
        progress: bool,
    ) -> None:
        """Stage imagery and run COLMAP sparse reconstruction."""
        images_dir = workspace / "images"
        staged = self._stage_images(images_dir, sequences, imagery_root=imagery_root, progress=progress)
        if staged < 2:
            raise COLMAPUnavailable(
                f"COLMAP needs at least two staged images under {images_dir}; staged {staged}."
            )
        database = workspace / "database.db"
        sparse_root = workspace / "sparse"
        sparse_root.mkdir(parents=True, exist_ok=True)
        timeout = int(os.getenv("COLMAP_TIMEOUT_SEC", "7200"))
        use_gpu = "1" if self.config.use_gpu else "0"
        commands = [
            [
                "feature_extractor",
                "--database_path",
                str(database),
                "--image_path",
                str(images_dir),
                "--ImageReader.single_camera",
                "0",
                "--SiftExtraction.use_gpu",
                use_gpu,
                "--SiftExtraction.num_threads",
                str(self.config.threads),
            ],
            [
                "sequential_matcher",
                "--database_path",
                str(database),
                "--SiftMatching.use_gpu",
                use_gpu,
            ],
            [
                "mapper",
                "--database_path",
                str(database),
                "--image_path",
                str(images_dir),
                "--output_path",
                str(sparse_root),
                "--Mapper.num_threads",
                str(self.config.threads),
            ],
        ]
        # GPU-specific enhancements: more features + explicit GPU index
        if self.config.use_gpu:
            commands[0].extend([
                "--SiftExtraction.max_num_features", "8192",
                "--SiftExtraction.gpu_index", self.config.gpu_index,
            ])
            commands[1].extend([
                "--SiftMatching.gpu_index", self.config.gpu_index,
            ])
        for command in commands:
            try:
                self._run_colmap(command, workspace=workspace, timeout=timeout)
            except subprocess.CalledProcessError as exc:
                raise COLMAPUnavailable(f"COLMAP command failed: {' '.join(command)}") from exc
            except subprocess.TimeoutExpired as exc:
                raise COLMAPUnavailable(f"COLMAP command timed out: {' '.join(command)}") from exc

    def _stage_images(
        self,
        images_dir: Path,
        sequences: Mapping[str, Iterable[FrameMeta]],
        *,
        imagery_root: Optional[Path | str],
        progress: bool = False,
    ) -> int:
        images_dir.mkdir(parents=True, exist_ok=True)
        staged = 0
        for frames in _progress_iter(sequences.values(), "COLMAP stage images", progress):
            for frame in frames:
                src = _find_cached_image(frame, imagery_root)
                if src is None:
                    continue
                dest = images_dir / f"{frame.image_id}{src.suffix.lower()}"
                if not dest.exists():
                    if self._using_docker():
                        shutil.copy2(src, dest)
                    else:
                        try:
                            os.symlink(src.resolve(), dest)
                        except OSError:
                            shutil.copy2(src, dest)
                staged += 1
        return staged

    def _convert_model_to_text(self, sparse_dir: Path, txt_dir: Path) -> None:
        if not sparse_dir.exists():
            raise COLMAPUnavailable(f"COLMAP sparse output missing: {sparse_dir}")
        txt_dir.mkdir(parents=True, exist_ok=True)
        command = [
            "model_converter",
            "--input_path",
            str(sparse_dir),
            "--output_path",
            str(txt_dir),
            "--output_type",
            "TXT",
        ]
        try:
            self._run_colmap(
                command,
                workspace=sparse_dir.parents[1],
                timeout=int(os.getenv("COLMAP_TIMEOUT_SEC", "7200")),
            )
        except subprocess.CalledProcessError as exc:
            raise COLMAPUnavailable("COLMAP model_converter failed") from exc
        except subprocess.TimeoutExpired as exc:
            raise COLMAPUnavailable("COLMAP model_converter timed out") from exc
        required = [txt_dir / "cameras.txt", txt_dir / "images.txt", txt_dir / "points3D.txt"]
        missing = [path for path in required if not path.exists()]
        if missing:
            raise COLMAPUnavailable(f"COLMAP text export missing files: {missing}")

    def _using_docker(self) -> bool:
        return shutil.which(self.colmap_cmd) is None and bool(os.getenv("COLMAP_DOCKER_IMAGE"))

    def _docker_image_available(self) -> bool:
        image = os.getenv("COLMAP_DOCKER_IMAGE")
        if not image or shutil.which("docker") is None:
            return False
        try:
            subprocess.run(
                ["docker", "image", "inspect", image],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=10,
            )
            return True
        except Exception:
            return False

    def _run_colmap(self, args: list[str], *, workspace: Path, timeout: int) -> None:
        if self._using_docker():
            image = os.getenv("COLMAP_DOCKER_IMAGE")
            if not image:
                raise COLMAPUnavailable("COLMAP_DOCKER_IMAGE is not set")
            command = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{workspace}:{workspace}",
                "-w",
                str(workspace),
                image,
                "colmap",
                *args,
            ]
        else:
            command = [self.colmap_cmd, *args]
        log_path = workspace / "colmap_commands.log"
        with log_path.open("ab") as stream:
            stream.write(("\n$ " + " ".join(command) + "\n").encode("utf8", errors="replace"))
            stream.flush()
            subprocess.run(
                command,
                cwd=str(workspace),
                check=True,
                timeout=timeout,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )

    # ------------------------------------------------------------------
    def _load_fixture(
        self,
        fixture_root: Path,
        sequences: Mapping[str, Iterable[FrameMeta]],
    ) -> Dict[str, ReconstructionResult]:
        """Load a sparse reconstruction exported via `colmap model_converter --output_type TXT`."""
        if fixture_root.is_dir():
            cameras_path = fixture_root / "cameras.txt"
            images_path = fixture_root / "images.txt"
            points_path = fixture_root / "points3D.txt"
        else:
            raise FileNotFoundError(
                f"COLMAP fixture {fixture_root} is not a directory containing cameras.txt/images.txt/points3D.txt"
            )

        for required in (cameras_path, images_path, points_path):
            if not required.exists():
                raise FileNotFoundError(f"Missing COLMAP fixture component: {required}")

        cameras = _parse_cameras(cameras_path)
        images = _parse_images(images_path)
        points = _parse_points(points_path)

        frames_by_id: Dict[str, FrameMeta] = {}
        for seq_id, frames in sequences.items():
            for frame in frames:
                frames_by_id[frame.image_id] = frame

        per_sequence_frames: Dict[str, list[FrameMeta]] = {}
        per_sequence_poses: Dict[str, Dict[str, Pose]] = {}
        image_id_to_seq: Dict[int, str] = {}
        image_id_to_frame: Dict[int, FrameMeta] = {}

        for image in images:
            frame = _resolve_frame(frames_by_id, image.name)
            if frame is None:
                log.debug("Skipping COLMAP image %s (no matching FrameMeta)", image.name)
                continue

            camera = cameras.get(image.camera_id)
            if camera is None:
                log.warning("COLMAP image %s references unknown camera %d", image.name, image.camera_id)
                continue

            updated_params = _merge_camera_params(frame.cam_params, camera)
            refined_frame = FrameMeta(
                image_id=frame.image_id,
                seq_id=frame.seq_id,
                captured_at_ms=frame.captured_at_ms,
                lon=frame.lon,
                lat=frame.lat,
                alt_ellip=frame.alt_ellip,
                camera_type=updated_params.pop("_camera_type"),
                cam_params=updated_params,
                quality_score=frame.quality_score,
                thumbnail_url=frame.thumbnail_url,
            )

            per_sequence_frames.setdefault(frame.seq_id, []).append(refined_frame)
            pose_map = per_sequence_poses.setdefault(frame.seq_id, {})
            pose_map[frame.image_id] = Pose(R=image.rotation_wc, t=image.center)
            image_id_to_seq[image.image_id] = frame.seq_id
            image_id_to_frame[image.image_id] = frame

        points_by_seq: Dict[str, list[list[float]]] = {seq: [] for seq in per_sequence_frames}

        for point in points:
            assigned = False
            for image_id in point.observations:
                seq_id = image_id_to_seq.get(image_id)
                if seq_id is None:
                    continue
                points_by_seq.setdefault(seq_id, []).append(point.xyz)
                assigned = True
                break
            if not assigned:
                log.debug("COLMAP point %d has no known observations; skipping", point.point_id)

        results: Dict[str, ReconstructionResult] = {}
        for seq_id, frames in per_sequence_frames.items():
            frames_sorted = sorted(frames, key=lambda f: f.captured_at_ms)
            poses = per_sequence_poses.get(seq_id, {})
            points_list = points_by_seq.get(seq_id, [])
            points_xyz = (
                np.asarray(points_list, dtype=np.float32)
                if points_list
                else np.zeros((0, 3), dtype=np.float32)
            )
            metadata = {
                "fixture": str(fixture_root),
                "point_count": int(points_xyz.shape[0]),
                "cameras_refined": False,
                "coordinate_frame": "enu",
            }
            results[seq_id] = ReconstructionResult(
                seq_id=seq_id,
                frames=frames_sorted,
                poses=poses,
                points_xyz=points_xyz,
                source="colmap",
                metadata=metadata,
            )

        return results


# ----------------------------------------------------------------------
# Parsing helpers


def _parse_cameras(path: Path) -> Dict[int, _ColmapCamera]:
    cameras: Dict[int, _ColmapCamera] = {}
    for line in _iter_data_lines(path):
        tokens = line.split()
        if len(tokens) < 5:
            continue
        camera_id = int(tokens[0])
        model = tokens[1]
        width = int(tokens[2])
        height = int(tokens[3])
        params = [float(tok) for tok in tokens[4:]]
        cameras[camera_id] = _ColmapCamera(
            camera_id=camera_id,
            model=model,
            width=width,
            height=height,
            params=params,
        )
    return cameras


@dataclass
class _ColmapPoint:
    point_id: int
    xyz: list[float]
    observations: list[int]


def _parse_images(path: Path) -> list[_ColmapImage]:
    images: list[_ColmapImage] = []
    lines = list(_iter_data_lines(path, keep_blank=True))
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        idx += 1
        if not line:
            continue
        tokens = line.split()
        if len(tokens) < 9:
            continue
        image_id = int(tokens[0])
        qw, qx, qy, qz = map(float, tokens[1:5])
        tx, ty, tz = map(float, tokens[5:8])
        camera_id = int(tokens[8])
        name = tokens[9] if len(tokens) > 9 else str(image_id)
        R_cw = _quat_to_matrix((qw, qx, qy, qz))
        R_wc = R_cw.T
        t_cw = np.array([tx, ty, tz], dtype=np.float64)
        center = -R_wc @ t_cw
        images.append(
            _ColmapImage(
                image_id=image_id,
                camera_id=camera_id,
                name=name,
                rotation_wc=R_wc,
                center=center,
            )
        )
        if idx < len(lines):
            idx += 1  # Skip observation line
    return images


def _parse_points(path: Path) -> list[_ColmapPoint]:
    points: list[_ColmapPoint] = []
    for line in _iter_data_lines(path):
        tokens = line.split()
        if len(tokens) < 7:
            continue
        point_id = int(tokens[0])
        x, y, z = map(float, tokens[1:4])
        track_tokens = tokens[8:]
        observations: list[int] = []
        for obs_idx in range(0, len(track_tokens), 2):
            try:
                observations.append(int(track_tokens[obs_idx]))
            except (ValueError, IndexError):
                continue
        points.append(_ColmapPoint(point_id=point_id, xyz=[x, y, z], observations=observations))
    return points


def _iter_data_lines(path: Path, *, keep_blank: bool = False):
    with path.open("r", encoding="utf8") as handle:
        for raw in handle:
            line = raw.strip("\n")
            if not line and not keep_blank:
                continue
            stripped = line.strip()
            if not stripped.startswith("#"):
                yield line if keep_blank else stripped


def _find_cached_image(frame: FrameMeta, imagery_root: Optional[Path | str]) -> Optional[Path]:
    roots: list[Path] = []
    if imagery_root is not None:
        roots.append(Path(imagery_root))
    roots.append(Path(constants.MAPILLARY_CACHE_ROOT) / "imagery")
    stems = [f"{frame.image_id}_{constants.MAPILLARY_DEFAULT_IMAGE_RES}", frame.image_id]
    suffixes = [".jpg", ".jpeg", ".png", ".bmp"]
    for root in roots:
        seq_dir = root / str(frame.seq_id)
        for stem in stems:
            for suffix in suffixes:
                candidate = seq_dir / f"{stem}{suffix}"
                if candidate.exists():
                    return candidate
    return None


def _progress_iter(items, desc: str, enabled: bool):
    if not enabled:
        return items
    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover - optional display dependency
        return items
    return tqdm(items, desc=desc, unit="seq")


def _resolve_frame(frames: Dict[str, FrameMeta], image_name: str) -> Optional[FrameMeta]:
    if image_name in frames:
        return frames[image_name]
    stem = Path(image_name).stem
    return frames.get(stem)


def _merge_camera_params(
    existing: Dict,
    camera: _ColmapCamera,
) -> Dict:
    params = existing.copy()
    params["width"] = camera.width
    params["height"] = camera.height
    model = camera.model.upper()
    camera_type = "perspective"

    if model in {"PINHOLE", "OPENCV", "FULL_OPENCV"}:
        fx, fy, cx, cy = camera.params[:4]
        params["focal_px"] = float((fx + fy) * 0.5)
        params["focal"] = float(params["focal_px"] / max(camera.width, camera.height))
        params["principal_point"] = [float(cx / camera.width), float(cy / camera.height)]
        params["fx_px"] = float(fx)
        params["fy_px"] = float(fy)
        params["cx_px"] = float(cx)
        params["cy_px"] = float(cy)
        if len(camera.params) >= 8:
            params["k1"] = float(camera.params[4])
            params["k2"] = float(camera.params[5])
            params["p1"] = float(camera.params[6])
            params["p2"] = float(camera.params[7])
    elif model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        f = camera.params[0]
        cx = camera.params[1] if len(camera.params) > 1 else camera.width * 0.5
        cy = camera.params[2] if len(camera.params) > 2 else camera.height * 0.5
        params["focal_px"] = float(f)
        params["focal"] = float(f / max(camera.width, camera.height))
        params["principal_point"] = [float(cx / camera.width), float(cy / camera.height)]
        params["cx_px"] = float(cx)
        params["cy_px"] = float(cy)
        params["fx_px"] = float(f)
        params["fy_px"] = float(f)
        if model in {"SIMPLE_RADIAL", "RADIAL"} and len(camera.params) >= 4:
            params["k1"] = float(camera.params[3])
        if model == "RADIAL" and len(camera.params) >= 5:
            params["k2"] = float(camera.params[4])
    elif model in {"FISHEYE", "FISHEYE_POLY", "OPENCV_FISHEYE"}:
        camera_type = "fisheye"
        fx, fy, cx, cy = camera.params[:4]
        params["focal_px"] = float((fx + fy) * 0.5)
        params["focal"] = float(params["focal_px"] / max(camera.width, camera.height))
        params["principal_point"] = [float(cx / camera.width), float(cy / camera.height)]
        params["fx_px"] = float(fx)
        params["fy_px"] = float(fy)
        params["cx_px"] = float(cx)
        params["cy_px"] = float(cy)
    else:
        log.debug("Unhandled COLMAP camera model '%s'; keeping existing intrinsics", model)

    params["_camera_type"] = camera_type
    return params
