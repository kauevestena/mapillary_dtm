from pathlib import Path

p = Path("tests/test_geometry_scaffolding.py")
content = """from __future__ import annotations

import sys
import types

import numpy as np

from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
if "dtm_from_mapillary" not in sys.modules:
    pkg = types.ModuleType("dtm_from_mapillary")
    pkg.__path__ = [str(ROOT)]
    sys.modules["dtm_from_mapillary"] = pkg
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dtm_from_mapillary.common_core import FrameMeta, Pose, ReconstructionResult
from dtm_from_mapillary.geom.sfm_opensfm import run as run_opensfm
from dtm_from_mapillary.geom.sfm_colmap import run as run_colmap
from dtm_from_mapillary.geom.vo_simplified import run as run_vo
from tests.sample_loader import get_sample_frames


def test_geometry_scaffolds_produce_results():
    seqs, imagery_root = get_sample_frames()
    
    # Try OpenSfM
    from dtm_from_mapillary.geom.opensfm_adapter import OpenSfMUnavailable
    try:
        opensfm = run_opensfm(seqs, imagery_root=imagery_root)
        seq_id = list(seqs.keys())[0]
        assert seq_id in opensfm
        assert isinstance(opensfm[seq_id], ReconstructionResult)
        assert opensfm[seq_id].points_xyz.shape[0] >= 12
    except OpenSfMUnavailable:
        pass

    # Try COLMAP
    from dtm_from_mapillary.geom.colmap_adapter import COLMAPUnavailable
    try:
        colmap = run_colmap(seqs, imagery_root=imagery_root)
        seq_id = list(seqs.keys())[0]
        assert colmap[seq_id].source == "colmap"
        assert colmap[seq_id].points_xyz.shape[0] >= 3
    except COLMAPUnavailable:
        pass

    # Try VO
    vo = run_vo(seqs, imagery_root=imagery_root)
    seq_id = list(seqs.keys())[0]
    assert vo[seq_id].source == "vo"
    assert vo[seq_id].points_xyz.size == 0
    assert vo[seq_id].metadata["mode"] == "opencv"
"""
p.write_text(content)
