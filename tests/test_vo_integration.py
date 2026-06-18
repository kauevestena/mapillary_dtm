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
from dtm_from_mapillary.geom.vo_simplified import run as run_vo
from tests.sample_loader import get_sample_frames

def test_vo_fails_without_imagery(tmp_path: Path):
    seqs, _ = get_sample_frames()
    with pytest.raises(RuntimeError, match="VO failed"):
        run_vo(seqs, imagery_root=tmp_path)

def test_vo_opencv_path():
    cv2 = pytest.importorskip("cv2")
    seqs, imagery_root = get_sample_frames()
    
    results = run_vo(
        seqs,
        imagery_root=imagery_root,
        min_inliers=10,
    )
    
    seq_id = list(seqs.keys())[0]
    frames = seqs[seq_id]
    
    assert seq_id in results
    result = results[seq_id]
    meta = result.metadata
    assert meta.get("mode") == "opencv"
    assert meta.get("pairs_processed") >= 3
    
    translations = np.array([result.poses[f.image_id].t for f in frames if f.image_id in result.poses])
    assert translations.shape[0] == len(frames)
    # Ensure motion accumulated along track
    assert not np.allclose(translations[0], translations[-1])

def test_vo_missing_imagery_fails(tmp_path: Path):
    seqs, _ = get_sample_frames()
    imagery_root = tmp_path / "imagery"
    imagery_root.mkdir(parents=True, exist_ok=True)

    with pytest.raises(RuntimeError, match="VO failed"):
        run_vo(seqs, imagery_root=imagery_root)
