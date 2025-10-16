from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if "dtm_from_mapillary" not in sys.modules:
    pkg = types.ModuleType("dtm_from_mapillary")
    pkg.__path__ = [str(ROOT)]
    sys.modules["dtm_from_mapillary"] = pkg
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dtm_from_mapillary import constants
from dtm_from_mapillary.common_core import FrameMeta
from dtm_from_mapillary.geom.vo_simplified import run as run_vo


def _build_frames(seq_id: str, n_frames: int = 3) -> list[FrameMeta]:
    return [
        FrameMeta(
            image_id=f"{seq_id}-frame-{idx}",
            seq_id=seq_id,
            captured_at_ms=1_700_000_000_000 + idx * 100,
            lon=-48.596644 + 0.0001 * idx,
            lat=-27.591363 + 0.0001 * idx,
            alt_ellip=10.0 + 0.1 * idx,
            camera_type="perspective",
            cam_params={
                "width": 1280,
                "height": 960,
                "focal": 0.85,
                "principal_point": [0.5, 0.5],
            },
            quality_score=0.9,
        )
        for idx in range(n_frames)
    ]


def test_vo_synthetic_force():
    frames = _build_frames("seqA")
    results = run_vo({"seqA": frames}, force_synthetic=True)
    assert "seqA" in results
    meta = results["seqA"].metadata
    assert meta.get("mode") == "synthetic"
    assert meta.get("scale") > 0


def test_vo_opencv_path(tmp_path: Path):
    cv2 = pytest.importorskip("cv2")

    frames = _build_frames("seqA", n_frames=4)
    imagery_root = tmp_path / "imagery"
    seq_dir = imagery_root / "seqA"
    seq_dir.mkdir(parents=True, exist_ok=True)

    for idx, frame in enumerate(frames):
        img = np.zeros((240, 320), dtype=np.uint8)
        center = (60 + idx * 15, 120)
        cv2.circle(img, center, 30, color=200, thickness=2)
        cv2.putText(
            img,
            str(idx),
            (20 + idx * 10, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            180,
            2,
            lineType=cv2.LINE_AA,
        )
        out_path = seq_dir / f"{frame.image_id}_{constants.MAPILLARY_DEFAULT_IMAGE_RES}.jpg"
        cv2.imwrite(str(out_path), img)

    results = run_vo(
        {"seqA": frames},
        imagery_root=imagery_root,
        force_synthetic=False,
        min_inliers=10,
    )

    assert "seqA" in results
    result = results["seqA"]
    meta = result.metadata
    assert meta.get("mode") == "opencv"
    assert meta.get("pairs_processed") >= 3
    translations = np.array([result.poses[f.image_id].t for f in frames if f.image_id in result.poses])
    assert translations.shape[0] == len(frames)
    # Ensure motion accumulated along track
    assert not np.allclose(translations[0], translations[-1])


def test_vo_missing_imagery_fallback(tmp_path: Path):
    frames = _build_frames("seqA", n_frames=3)
    imagery_root = tmp_path / "imagery"
    imagery_root.mkdir(parents=True, exist_ok=True)

    results = run_vo({"seqA": frames}, imagery_root=imagery_root, force_synthetic=False)
    assert "seqA" in results
    assert results["seqA"].metadata.get("mode") == "synthetic"
