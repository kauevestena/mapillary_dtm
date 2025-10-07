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

from dtm_from_mapillary.ingest.sequence_filter import filter_car_sequences
from dtm_from_mapillary.common_core import FrameMeta


def build_frame(image_id: str, seq: str, lon: float, lat: float, captured_ms: int, quality: float = 0.5,
                camera_type: str = "perspective") -> FrameMeta:
    return FrameMeta(
        image_id=image_id,
        seq_id=seq,
        captured_at_ms=captured_ms,
        lon=lon,
        lat=lat,
        alt_ellip=None,
        camera_type=camera_type,
        cam_params={},
        quality_score=quality,
    )


def test_filter_car_sequences_keeps_reasonable_speeds():
    frames = [
        build_frame("f0", "seq1", 0.0, 0.0, 0),
        build_frame("f1", "seq1", 0.00015, 0.00015, 1000),
        build_frame("f2", "seq1", 0.00030, 0.00030, 2000),
    ]
    seqs = {"seq1": frames}

    filtered = filter_car_sequences(seqs)
    assert "seq1" in filtered
    assert len(filtered["seq1"]) == len(frames)


def test_filter_car_sequences_drops_low_quality_and_speed():
    frames = [
        build_frame("f0", "seq2", 0.0, 0.0, 0, quality=0.1),  # quality too low
        build_frame("f1", "seq2", 0.0, 0.0, 1000),
        build_frame("f2", "seq2", 0.0, 0.0, 2000),
    ]
    seqs = {"seq2": frames}

    filtered = filter_car_sequences(seqs)
    assert "seq2" not in filtered


def test_filter_car_sequences_drops_wrong_camera_type():
    frames = [
        build_frame("f0", "seq3", 0.0, 0.0, 0, camera_type="unknown"),
        build_frame("f1", "seq3", 0.00015, 0.00015, 1000, camera_type="unknown"),
    ]

    filtered = filter_car_sequences({"seq3": frames})
    assert filtered == {}
