from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if "dtm_from_mapillary" not in sys.modules:
    pkg = types.ModuleType("dtm_from_mapillary")
    pkg.__path__ = [str(ROOT)]
    sys.modules["dtm_from_mapillary"] = pkg
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dtm_from_mapillary.common_core import FrameMeta
from dtm_from_mapillary.geom.opensfm_adapter import OpenSfMRunner
from dtm_from_mapillary.geom.sfm_opensfm import run as run_opensfm


def _sample_sequence() -> dict[str, list[FrameMeta]]:
    frames = [
        FrameMeta(
            image_id="img-1",
            seq_id="seq-1",
            captured_at_ms=0,
            lon=0.0,
            lat=0.0,
            alt_ellip=0.0,
            camera_type="perspective",
            cam_params={"width": 4000, "height": 3000},
            quality_score=0.9,
        ),
        FrameMeta(
            image_id="img-2",
            seq_id="seq-1",
            captured_at_ms=1000,
            lon=0.0,
            lat=0.0,
            alt_ellip=0.0,
            camera_type="perspective",
            cam_params={"width": 4000, "height": 3000},
            quality_score=0.85,
        ),
    ]
    return {"seq-1": frames}


def _fixture_path() -> Path:
    return ROOT / "qa" / "data" / "opensfm_fixture" / "reconstruction.json"


def test_opensfm_runner_fixture(tmp_path: Path) -> None:
    seqs = _sample_sequence()
    runner = OpenSfMRunner(workspace_root=tmp_path)
    results = runner.reconstruct(seqs, fixture_path=_fixture_path())

    assert "seq-1" in results
    recon = results["seq-1"]
    assert recon.metadata["fixture"].endswith("reconstruction.json")
    assert recon.points_xyz.shape == (2, 3)
    pose = recon.poses["img-1"]
    assert pose.R.shape == (3, 3)
    assert np.allclose(pose.R.T @ pose.R, np.eye(3), atol=1e-6)


def test_opensfm_run_uses_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    seqs = _sample_sequence()
    fixture = _fixture_path()
    monkeypatch.setenv("OPEN_SFM_FIXTURE", str(fixture))
    monkeypatch.delenv("OPEN_SFM_FORCE_SYNTHETIC", raising=False)

    results = run_opensfm(seqs)
    recon = results["seq-1"]
    assert recon.metadata and recon.metadata.get("fixture") == str(fixture)

    monkeypatch.delenv("OPEN_SFM_FIXTURE", raising=False)


def test_opensfm_run_falls_back_to_synthetic(monkeypatch: pytest.MonkeyPatch) -> None:
    seqs = _sample_sequence()
    monkeypatch.delenv("OPEN_SFM_FIXTURE", raising=False)
    monkeypatch.delenv("OPEN_SFM_FORCE_SYNTHETIC", raising=False)

    results = run_opensfm(seqs)
    recon = results["seq-1"]
    assert "fixture" not in (recon.metadata or {})
    assert recon.metadata.get("cameras_refined") is False
