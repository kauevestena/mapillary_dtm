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

from dtm_from_mapillary.common_core import FrameMeta
from dtm_from_mapillary.depth.monodepth import predict_depths, _DepthAdapter


from tests.sample_loader import get_sample_frames


class _StubAdapter(_DepthAdapter):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def predict(self, frame: FrameMeta):
        self.calls.append(frame.image_id)
        depth = np.full((10, 16), 5.0 + len(self.calls), dtype=np.float32)
        uncert = np.full_like(depth, 0.25)
        return depth, uncert


def test_monodepth_adapter_outputs(tmp_path: Path):
    seqs, _ = get_sample_frames()
    seq_id = list(seqs.keys())[0]
    frames = {seq_id: seqs[seq_id][:2]}
    adapter = _StubAdapter()
    results = predict_depths(frames, out_dir=tmp_path, adapter=adapter, force=True)

    frame_ids = [f.image_id for f in frames[seq_id]]
    assert adapter.calls == frame_ids
    seq_res = results[seq_id]
    depth = seq_res[frame_ids[0]]["depth"]
    uncert = seq_res[frame_ids[0]]["uncertainty"]
    assert depth.shape == (10, 16)
    assert np.allclose(uncert, 0.25)


class _NoneAdapter(_DepthAdapter):
    def predict(self, frame: FrameMeta):
        return None


def test_monodepth_adapter_fails_without_model(tmp_path: Path):
    seqs, _ = get_sample_frames()
    seq_id = list(seqs.keys())[0]
    frames = {seq_id: seqs[seq_id][:1]}
    with pytest.raises(RuntimeError, match="Monodepth prediction unavailable"):
        predict_depths(frames, out_dir=tmp_path, adapter=_NoneAdapter(), force=True)
