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
from dtm_from_mapillary.semantics.ground_masks import prepare


def build_frame(image_id="img", seq_id="seq", width=4000, height=3000):
    return FrameMeta(
        image_id=image_id,
        seq_id=seq_id,
        captured_at_ms=1234,
        lon=0.0,
        lat=0.0,
        alt_ellip=None,
        camera_type="perspective",
        cam_params={"width": width, "height": height},
        quality_score=0.9,
    )


def test_prepare_creates_masks(tmp_path):
    frame = build_frame("img-1")
    out = prepare({"seq": [frame]}, out_dir=tmp_path, backend="constant")

    mask_path = tmp_path / "img-1.npz"
    assert mask_path.exists()
    assert out == {"seq": [mask_path]}

    with np.load(mask_path) as data:
        prob = data["prob"]
        assert prob.shape[1] == 256
        assert prob.shape[0] > 150
        assert np.all(prob >= 0.0) and np.all(prob <= 1.0)


def test_prepare_uses_existing_cache(tmp_path):
    frame = build_frame("img-cache")
    mask_path = tmp_path / "img-cache.npz"
    existing = np.full((10, 10), 0.42, dtype=np.float32)
    np.savez_compressed(mask_path, prob=existing, image_id="img-cache", seq_id="seq")

    prepare({"seq": [frame]}, out_dir=tmp_path, backend="soft-horizon")

    with np.load(mask_path) as data:
        prob = data["prob"]
        assert prob.shape == (10, 10)
        assert np.isclose(prob.mean(), 0.42)


def test_prepare_force_overwrites(tmp_path):
    frame = build_frame("img-force")
    mask_path = tmp_path / "img-force.npz"
    np.savez_compressed(mask_path, prob=np.zeros((5, 5), dtype=np.float32))

    prepare({"seq": [frame]}, out_dir=tmp_path, backend="constant", force=True)

    with np.load(mask_path) as data:
        prob = data["prob"]
        assert prob.shape[0] > 5
        assert np.isclose(prob.mean(), 0.75, atol=1e-3)
