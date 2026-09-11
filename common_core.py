"""
Shared dataclasses and core math utilities.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
import numpy as np

@dataclass
class FrameMeta:
    image_id: str
    seq_id: str
    captured_at_ms: int
    lon: float
    lat: float
    alt_ellip: Optional[float]
    camera_type: str  # "perspective"|"fisheye"|"spherical"
    cam_params: Dict  # fx, fy, cx, cy, distortion, etc (OpenSfM-like)
    quality_score: Optional[float]
    thumbnail_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-friendly dict."""
        return {
            "image_id": self.image_id,
            "seq_id": self.seq_id,
            "captured_at_ms": int(self.captured_at_ms),
            "lon": float(self.lon),
            "lat": float(self.lat),
            "alt_ellip": float(self.alt_ellip) if self.alt_ellip is not None else None,
            "camera_type": self.camera_type,
            "cam_params": self.cam_params,
            "quality_score": float(self.quality_score) if self.quality_score is not None else None,
            "thumbnail_url": self.thumbnail_url,
        }

    def to_geojson_feature(self) -> Dict[str, Any]:
        """Return a GeoJSON Feature with a Point geometry at the GNSS camera position.

        The ``geometry`` uses WGS84 (lon, lat) coordinates.  Ellipsoidal altitude
        is included as the optional third coordinate when available.  All frame
        attributes are placed in ``properties`` so the file can be opened directly
        in GIS tools (QGIS, geojson.io, etc.).
        """
        coordinates: list = [float(self.lon), float(self.lat)]
        if self.alt_ellip is not None:
            coordinates.append(float(self.alt_ellip))
        return {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": coordinates,
            },
            "properties": {
                "image_id": self.image_id,
                "seq_id": self.seq_id,
                "captured_at_ms": int(self.captured_at_ms),
                "camera_type": self.camera_type,
                "cam_params": self.cam_params,
                "quality_score": float(self.quality_score) if self.quality_score is not None else None,
                "thumbnail_url": self.thumbnail_url,
            },
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "FrameMeta":
        if data.get("image_id") is None or data.get("seq_id") is None:
            raise ValueError("FrameMeta requires image_id and seq_id")
        return FrameMeta(
            image_id=str(data.get("image_id")),
            seq_id=str(data.get("seq_id")),
            captured_at_ms=int(data.get("captured_at_ms", 0)),
            lon=float(data.get("lon", 0.0)),
            lat=float(data.get("lat", 0.0)),
            alt_ellip=float(data["alt_ellip"]) if data.get("alt_ellip") is not None else None,
            camera_type=str(data.get("camera_type", "unknown")),
            cam_params=dict(data.get("cam_params", {})),
            quality_score=float(data["quality_score"]) if data.get("quality_score") is not None else None,
            thumbnail_url=data.get("thumbnail_url"),
        )

@dataclass
class Pose:
    """World-from-camera pose (R,t), in the raw reconstruction frame."""
    R: np.ndarray  # (3,3)
    t: np.ndarray  # (3,)


@dataclass
class ReconstructionResult:
    """Simplified reconstruction payload shared across geometry stacks."""

    seq_id: str
    frames: List[FrameMeta]
    poses: Dict[str, Pose]
    points_xyz: np.ndarray  # (N,3) ground/sparse points
    source: str
    metadata: Optional[Dict[str, Any]] = None
    coordinate_frame: str = "reconstruction"

    def __post_init__(self) -> None:
        if self.points_xyz.ndim != 2 or self.points_xyz.shape[1] != 3:
            raise ValueError(
                f"points_xyz must have shape (N, 3); received {self.points_xyz.shape}"
            )
        if not np.isfinite(self.points_xyz).all():
            raise ValueError("points_xyz contains non-finite values")

        for image_id, pose in self.poses.items():
            if pose.R.shape != (3, 3) or pose.t.shape != (3,):
                raise ValueError(f"Pose for {image_id} has invalid shape")
            if not np.isfinite(pose.R).all() or not np.isfinite(pose.t).all():
                raise ValueError(f"Pose for {image_id} contains non-finite values")

        allowed_frames = {"reconstruction"}
        if self.coordinate_frame not in allowed_frames:
            raise ValueError(
                f"Unsupported coordinate frame '{self.coordinate_frame}'. "
                f"Allowed: {sorted(allowed_frames)}"
            )

        if self.metadata is None:
            self.metadata = {}
        elif not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict if provided")

        frame_tag = self.metadata.get("coordinate_frame")
        if frame_tag is None:
            self.metadata["coordinate_frame"] = self.coordinate_frame
        elif frame_tag != self.coordinate_frame:
            raise ValueError(
                "metadata coordinate_frame mismatch: "
                f"{frame_tag} vs {self.coordinate_frame}"
            )

