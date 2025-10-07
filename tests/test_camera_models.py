from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if "dtm_from_mapillary" not in sys.modules:
    pkg = types.ModuleType("dtm_from_mapillary")
    pkg.__path__ = [str(ROOT)]
    sys.modules["dtm_from_mapillary"] = pkg
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dtm_from_mapillary.ingest.camera_models import make_opensfm_model
from dtm_from_mapillary.common_core import FrameMeta


def build_frame(cam_type="perspective", cam_params=None):
    return FrameMeta(
        image_id="id",
        seq_id="seq",
        captured_at_ms=0,
        lon=0.0,
        lat=0.0,
        alt_ellip=None,
        camera_type=cam_type,
        cam_params=cam_params or {},
        quality_score=None,
    )


def test_make_opensfm_model_perspective():
    frame = build_frame(
        cam_params={
            "width": 4000,
            "height": 3000,
            "fx": 2800.0,
            "fy": 2805.0,
            "cx": 2000.0,
            "cy": 1500.0,
            "k1": 0.1,
            "p1": 0.001,
        }
    )
    model = make_opensfm_model(frame)

    assert model["projection_type"] == "perspective"
    assert model["width"] == 4000 and model["height"] == 3000
    assert abs(model["focal"] - (2800.0 / 4000.0)) < 1e-6
    assert model["principal_point"] == [0.5, 0.5]
    assert model["k1"] == 0.1 and model["p1"] == 0.001


def test_make_opensfm_model_spherical_defaults():
    frame = build_frame(cam_type="spherical", cam_params={})
    model = make_opensfm_model(frame)
    assert model["projection_type"] == "spherical"
