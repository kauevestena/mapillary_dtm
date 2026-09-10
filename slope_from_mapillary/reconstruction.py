"""Read real SfM points AND their measured image observations.

Coordinates remain in the reconstruction frame until explicitly referenced.
No GPS elevations, camera heights, monocular depths or trajectory points enter
the surface measurements. Disconnected reconstructions are never concatenated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class Shot:
    name: str
    center: np.ndarray
    rotation_cw: np.ndarray
    width: int
    height: int


@dataclass
class Point:
    id: str
    xyz: np.ndarray
    error_px: float
    observations: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass
class Reconstruction:
    shots: dict[str, Shot]
    points: dict[str, Point]
    source: str

    def validate(self) -> "Reconstruction":
        if not self.shots or not self.points:
            raise ValueError("Reconstruction has no registered images or 3D points")
        for shot in self.shots.values():
            if shot.width <= 0 or shot.height <= 0:
                raise ValueError(f"Missing image dimensions: {shot.name}")
            if not np.isfinite(shot.center).all() or not np.isfinite(shot.rotation_cw).all():
                raise ValueError(f"Non-finite pose: {shot.name}")
        for point in self.points.values():
            if point.xyz.shape != (3,) or not np.isfinite(point.xyz).all():
                raise ValueError(f"Invalid 3D point: {point.id}")
        return self


def _data_lines(path: Path):
    for line in path.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            yield line.split()


def load_opensfm(path: Path, tracks: Path | None = None, component: int = 0) -> Reconstruction:
    """OpenSfM axis-angle poses and normalized track coordinates.

    Track coordinates are normalized by max(width, height), centered on the
    image. The scale/color columns vary between tracks-file versions; neither
    affects the first five columns used here.
    """
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, list) or not 0 <= component < len(payload):
        raise ValueError("Select an existing OpenSfM component index")
    model = payload[component]
    shots = {}
    for name, shot in model.get("shots", {}).items():
        camera = model["cameras"][shot["camera"]]
        vector = np.asarray(shot["rotation"], dtype=float)
        if vector.shape != (3,):
            raise ValueError("OpenSfM rotation must be a three-element axis-angle vector")
        rotation = Rotation.from_rotvec(vector).as_matrix()
        shots[name] = Shot(name, -rotation.T @ np.asarray(shot["translation"]), rotation,
                           int(camera["width"]), int(camera["height"]))
    # OpenSfM reprojection errors are in normalized image coordinates. When
    # images differ in size, use the largest observing image below.
    points = {str(key): Point(str(key), np.asarray(value["coordinates"], dtype=float),
                             float(value.get("reprojection_error", float("inf"))))
              for key, value in model.get("points", {}).items()}
    if tracks is not None:
        for row in _data_lines(Path(tracks)):
            if row[0].startswith("OPENSFM_TRACKS_VERSION"):
                continue
            if len(row) < 5:
                raise ValueError("Invalid OpenSfM tracks row")
            shot, point = shots.get(row[0]), points.get(row[1])
            if shot is None or point is None:
                continue
            dimension = max(shot.width, shot.height)
            pixel = (float(row[3]) * dimension + shot.width / 2,
                     float(row[4]) * dimension + shot.height / 2)
            if shot.name in point.observations:
                raise ValueError(f"Duplicate observation for point {point.id} in {shot.name}")
            point.observations[shot.name] = pixel
    for point in points.values():
        size = max((max(shots[name].width, shots[name].height)
                    for name in point.observations), default=1)
        point.error_px *= size
    return Reconstruction(shots, points, "opensfm").validate()


def load_colmap(path: Path) -> Reconstruction:
    """COLMAP text export, including the actual POINT2D_IDX observations."""
    path = Path(path)
    dimensions = {int(row[0]): (int(row[2]), int(row[3]))
                  for row in _data_lines(path / "cameras.txt")}
    # Preserve empty second lines: an image can have zero measured keypoints.
    lines = iter(line for line in (path / "images.txt").read_text().splitlines()
                 if not line.lstrip().startswith("#"))
    shots, image_rows = {}, {}
    for line in lines:
        if not line.strip():
            continue
        row = line.split(maxsplit=9)
        if len(row) != 10:
            raise ValueError("Invalid COLMAP image pose row")
        quaternion = np.asarray(row[1:5], dtype=float)
        if not np.isclose(np.linalg.norm(quaternion), 1, atol=1e-4):
            raise ValueError("COLMAP pose quaternion must have unit length")
        rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
        width, height = dimensions[int(row[8])]
        name = row[9]
        shots[name] = Shot(name, -rotation.T @ np.asarray(row[5:8], dtype=float),
                           rotation, width, height)
        try:
            values = next(lines).split()
        except StopIteration as exc:
            raise ValueError("Missing COLMAP image observation row") from exc
        if len(values) % 3:
            raise ValueError("Invalid COLMAP observation triples")
        image_rows[int(row[0])] = (name, [tuple(values[i:i+3])
                                        for i in range(0, len(values), 3)])
    points = {}
    for row in _data_lines(path / "points3D.txt"):
        if len(row) < 8 or (len(row) - 8) % 2:
            raise ValueError("Invalid COLMAP point track")
        point = Point(row[0], np.asarray(row[1:4], dtype=float), float(row[7]))
        for offset in range(8, len(row), 2):
            name, observations = image_rows[int(row[offset])]
            index = int(row[offset + 1])
            if not 0 <= index < len(observations):
                raise ValueError("COLMAP point references a missing keypoint")
            x, y, point_id = observations[index]
            if point_id != point.id or name in point.observations:
                raise ValueError("Inconsistent COLMAP point/image observation")
            point.observations[name] = (float(x), float(y))
        points[point.id] = point
    return Reconstruction(shots, points, "colmap").validate()


def load(spec: dict, root: Path) -> Reconstruction:
    if spec["format"] == "opensfm":
        tracks = (root / spec["tracks"]).resolve() if spec.get("tracks") else None
        return load_opensfm((root / spec["path"]).resolve(), tracks, spec.get("component", 0))
    if spec["format"] == "colmap":
        return load_colmap((root / spec["path"]).resolve())
    raise ValueError("Reconstruction format must be opensfm or colmap (text export)")


def triangulation_angle(point: Point, shots: dict[str, Shot]) -> float:
    rays = np.asarray([point.xyz - shots[name].center for name in point.observations])
    if len(rays) < 2:
        return 0.0
    lengths = np.linalg.norm(rays, axis=1)
    if np.any(lengths < 1e-12):
        return 0.0
    rays /= lengths[:, None]
    # Baseline angle, not the number of projections into arbitrary cameras.
    dots = np.clip(rays @ rays.T, -1, 1)
    return float(np.degrees(np.arccos(dots.min())))
