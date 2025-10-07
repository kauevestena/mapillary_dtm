from __future__ import annotations

import json
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

from dtm_from_mapillary.common_core import FrameMeta
from dtm_from_mapillary.geom.anchors import find_anchors
from dtm_from_mapillary.geom.height_solver import solve_scale_and_h
from dtm_from_mapillary.geom.sfm_opensfm import run as run_opensfm
from dtm_from_mapillary.geom.sfm_colmap import run as run_colmap
from dtm_from_mapillary.geom.vo_simplified import run as run_vo


def build_frames(seq_id: str) -> list[FrameMeta]:
    return [
        FrameMeta(
            image_id=f"{seq_id}-frame-{i}",
            seq_id=seq_id,
            captured_at_ms=1_700_000_000_000 + i * 200,
            lon=-48.5966 + 0.00012 * i,
            lat=-27.5913 + 0.00012 * i,
            alt_ellip=12.0 + 0.05 * i,
            camera_type="perspective",
            cam_params={"width": 4000, "height": 3000},
            quality_score=0.8,
        )
        for i in range(5)
    ]


def test_height_solver_with_sample_anchors(tmp_path):
    seqs = {"sample-seq": build_frames("sample-seq")}

    opensfm = run_opensfm(seqs)
    colmap = run_colmap(seqs)
    vo = run_vo(seqs)

    sample_file = tmp_path / "anchors.json"
    sample_file.write_text(
        json.dumps(
            {
                "anchors": [
                    {
                        "seq_id": "sample-seq",
                        "anchor_id": "anc-1",
                        "lon": -48.5965,
                        "lat": -27.5912,
                        "alt_ellip": 16.0,
                        "height_m": 4.0,
                        "diameter_m": 0.3,
                        "source": "unit-test",
                        "observations": [
                            {"image_id": "sample-seq-frame-0", "px": 2000.0, "py": 1200.0, "prob": 0.9},
                            {"image_id": "sample-seq-frame-1", "px": 2050.0, "py": 1180.0, "prob": 0.88},
                        ],
                    }
                ]
            },
            indent=2,
        )
    )

    anchors = find_anchors(
        seqs,
        cache_dir=tmp_path / "cache",
        sample_path=sample_file,
    )
    assert anchors
    assert anchors[0].seq_id == "sample-seq"

    scales, heights = solve_scale_and_h(opensfm, colmap, vo, anchors, seqs)
    assert "sample-seq" in scales and "sample-seq" in heights
    assert 0.5 < scales["sample-seq"] <= 4.0
    assert 1.0 <= heights["sample-seq"] <= 3.0
