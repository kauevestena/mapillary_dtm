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
from tests.sample_loader import get_sample_frames





def test_make_opensfm_model_perspective():
    seqs, _ = get_sample_frames()
    seq_id = list(seqs.keys())[0]
    frame = seqs[seq_id][0]
    
    model = make_opensfm_model(frame)

    assert model["projection_type"] == frame.camera_type


def test_make_opensfm_model_spherical_defaults():
    seqs, _ = get_sample_frames()
    seq_id = list(seqs.keys())[0]
    frame = seqs[seq_id][0]
    # Simulate a spherical camera
    frame.camera_type = "spherical"
    frame.cam_params = {}
    
    model = make_opensfm_model(frame)
    assert model["projection_type"] == "spherical"
