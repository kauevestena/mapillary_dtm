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
from dtm_from_mapillary.semantics.curb_edge_lane import CurbLine, extract_curbs_and_lanes


def build_frame(image_id: str, width=100, height=80):
    return FrameMeta(
        image_id=image_id,
        seq_id="seq1",
        captured_at_ms=0,
        lon=0.0,
        lat=0.0,
        alt_ellip=None,
        camera_type="perspective",
        cam_params={"width": width, "height": height},
        quality_score=0.8,
    )


def test_extract_curbs_and_lanes(tmp_path):
    frame = build_frame("img-curb")
    prob = np.linspace(0.0, 1.0, frame.cam_params["height"], dtype=np.float32)[:, None]
    prob = np.repeat(prob, frame.cam_params["width"], axis=1)
    np.savez_compressed(tmp_path / "img-curb.npz", prob=prob)

    curbs = extract_curbs_and_lanes({"seq1": [frame]}, mask_dir=tmp_path)
    assert "seq1" in curbs
    assert len(curbs["seq1"]) == 1
    line = curbs["seq1"][0]
    assert isinstance(line, CurbLine)
    assert len(line.xy_norm) > 50
    xs, ys = zip(*line.xy_norm)
    assert min(xs) > 0.0 and max(xs) <= 1.0
    assert min(ys) >= 0.0 and max(ys) <= 1.0
    assert line.confidence > 0.3


def test_extract_curbs_and_lanes_handles_missing(tmp_path):
    frame = build_frame("missing")
    curbs = extract_curbs_and_lanes({"seq1": [frame]}, mask_dir=tmp_path)
    assert curbs == {}
