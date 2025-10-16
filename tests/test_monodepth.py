from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if "dtm_from_mapillary" not in sys.modules:
    pkg = types.ModuleType("dtm_from_mapillary")
    pkg.__path__ = [str(ROOT)]
    sys.modules["dtm_from_mapillary"] = pkg
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dtm_from_mapillary.common_core import FrameMeta
from dtm_from_mapillary.depth.monodepth import predict_depths, _DepthAdapter


def _build_frames(seq_id: str, n_frames: int = 3) -> list[FrameMeta]:
    return [
        FrameMeta(
            image_id=f"{seq_id}-frame-{idx}",
            seq_id=seq_id,
            captured_at_ms=1_700_000_000_000 + idx * 50,
            lon=-48.596 + 0.0001 * idx,
            lat=-27.591 + 0.0001 * idx,
            alt_ellip=12.0,
            camera_type="perspective",
            cam_params={"width": 800, "height": 600, "focal": 0.9},
            quality_score=0.8,
        )
        for idx in range(n_frames)
    ]


class _StubAdapter(_DepthAdapter):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def predict(self, frame: FrameMeta):
        self.calls.append(frame.image_id)
        depth = np.full((10, 16), 5.0 + len(self.calls), dtype=np.float32)
        uncert = np.full_like(depth, 0.25)
        return depth, uncert


def test_monodepth_adapter_outputs(tmp_path: Path):
    frames = {"seqA": _build_frames("seqA", n_frames=2)}
    adapter = _StubAdapter()
    results = predict_depths(frames, out_dir=tmp_path, adapter=adapter, force=True)

    assert adapter.calls == ["seqA-frame-0", "seqA-frame-1"]
    seq_res = results["seqA"]
    depth = seq_res["seqA-frame-0"]["depth"]
    uncert = seq_res["seqA-frame-0"]["uncertainty"]
    assert depth.shape == (10, 16)
    assert np.allclose(uncert, 0.25)


class _NoneAdapter(_DepthAdapter):
    def predict(self, frame: FrameMeta):
        return None


def test_monodepth_adapter_fallback(tmp_path: Path):
    frames = {"seqA": _build_frames("seqA", n_frames=1)}
    results = predict_depths(frames, out_dir=tmp_path, adapter=_NoneAdapter(), force=True)
    depth = results["seqA"]["seqA-frame-0"]["depth"]
    assert depth.ndim == 2
    assert depth.size > 0
