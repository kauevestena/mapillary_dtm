from __future__ import annotations

import sys
import types

import numpy as np

from pathlib import Path

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


def _build_frames(seq_id: str) -> list[FrameMeta]:
    return [
        FrameMeta(
            image_id=f"{seq_id}-frame-{i}",
            seq_id=seq_id,
            captured_at_ms=1_700_000_000_000 + i * 100,
            lon=-48.596644 + 0.0001 * i,
            lat=-27.591363 + 0.0001 * i,
            alt_ellip=10.0 + 0.1 * i,
            camera_type="perspective",
            cam_params={"width": 4000, "height": 3000},
            quality_score=0.9,
        )
        for i in range(4)
    ]


def test_geometry_scaffolds_produce_results():
    seqs = {"seqA": _build_frames("seqA")}

    opensfm = run_opensfm(seqs)
    colmap = run_colmap(seqs)
    vo = run_vo(seqs)

    assert set(opensfm.keys()) == {"seqA"}
    assert isinstance(opensfm["seqA"], ReconstructionResult)
    assert opensfm["seqA"].points_xyz.shape[0] >= 12

    assert colmap["seqA"].source == "colmap"
    assert colmap["seqA"].points_xyz.shape[0] >= 12
    # Ensure the solutions are not identical (independent stacks)
    assert not np.allclose(opensfm["seqA"].points_xyz, colmap["seqA"].points_xyz)

    assert vo["seqA"].source == "vo"
    assert vo["seqA"].points_xyz.size == 0
    assert vo["seqA"].metadata["scale"] > 0

    # Poses should contain each frame and be proper Pose objects
    for recon in (opensfm["seqA"], colmap["seqA"], vo["seqA"]):
        assert set(recon.poses.keys()) == {frame.image_id for frame in seqs["seqA"]}
        sample_pose = recon.poses[next(iter(recon.poses))]
        assert isinstance(sample_pose, Pose)
        assert sample_pose.R.shape == (3, 3)
        assert sample_pose.t.shape == (3,)
