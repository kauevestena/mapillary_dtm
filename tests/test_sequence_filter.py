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


def build_frame(
    image_id: str,
    seq: str,
    lon: float,
    lat: float,
    captured_ms: int,
    quality: float = 0.5,
    camera_type: str = "perspective",
) -> FrameMeta:
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
    """Test that sequences with max speeds in car range are kept."""
    # Create frames with speeds that reach ~50 km/h
    # 0.00045 deg ≈ 50m at equator, over 1 second ≈ 50 m/s ≈ 180 km/h
    # 0.00015 deg ≈ 17m at equator, over 1 second ≈ 17 m/s ≈ 61 km/h
    frames = [
        build_frame("f0", "seq1", 0.0, 0.0, 0),
        build_frame("f1", "seq1", 0.00015, 0.00015, 1000),  # ~61 km/h
        build_frame("f2", "seq1", 0.00030, 0.00030, 2000),  # ~61 km/h
    ]
    seqs = {"seq1": frames}

    filtered = filter_car_sequences(seqs)
    assert "seq1" in filtered, "Sequence with car-speed should be kept"
    assert len(filtered["seq1"]) == len(frames)


def test_filter_car_sequences_drops_too_slow():
    """Test that sequences with max speeds below threshold are rejected."""
    # Create frames with speeds that only reach ~10 km/h (pedestrian/bike)
    # 0.00003 deg ≈ 3.3m at equator, over 1 second ≈ 3.3 m/s ≈ 12 km/h
    frames = [
        build_frame("f0", "seq2", 0.0, 0.0, 0),
        build_frame("f1", "seq2", 0.00003, 0.00003, 1000),  # ~12 km/h
        build_frame("f2", "seq2", 0.00006, 0.00006, 2000),  # ~12 km/h
    ]
    seqs = {"seq2": frames}

    filtered = filter_car_sequences(seqs, min_speed_kmh=40.0, max_speed_kmh=120.0)
    assert "seq2" not in filtered, "Slow sequence should be rejected"


def test_filter_car_sequences_drops_too_fast():
    """Test that sequences with max speeds above threshold are rejected."""
    # Create frames with speeds that reach ~150 km/h (highway)
    # 0.00075 deg ≈ 83m at equator, over 1 second ≈ 83 m/s ≈ 300 km/h
    frames = [
        build_frame("f0", "seq3", 0.0, 0.0, 0),
        build_frame("f1", "seq3", 0.00075, 0.00075, 1000),  # ~300 km/h
        build_frame("f2", "seq3", 0.00150, 0.00150, 2000),  # ~300 km/h
    ]
    seqs = {"seq3": frames}

    filtered = filter_car_sequences(seqs, min_speed_kmh=40.0, max_speed_kmh=120.0)
    assert "seq3" not in filtered, "Extremely fast sequence should be rejected"


def test_filter_car_sequences_drops_low_quality_frames():
    """Test that individual frames with low quality are filtered out."""
    frames = [
        build_frame("f0", "seq4", 0.0, 0.0, 0, quality=0.1),  # quality too low
        build_frame("f1", "seq4", 0.00015, 0.00015, 1000, quality=0.5),
        build_frame("f2", "seq4", 0.00030, 0.00030, 2000, quality=0.5),
    ]
    seqs = {"seq4": frames}

    filtered = filter_car_sequences(seqs)
    # Sequence should pass speed test, but low-quality frame should be dropped
    if "seq4" in filtered:
        assert len(filtered["seq4"]) == 2, "Low quality frame should be filtered"


def test_filter_car_sequences_drops_wrong_camera_type():
    """Test that frames with unsupported camera types are filtered."""
    frames = [
        build_frame("f0", "seq5", 0.0, 0.0, 0, camera_type="unknown"),
        build_frame("f1", "seq5", 0.00015, 0.00015, 1000, camera_type="unknown"),
    ]

    filtered = filter_car_sequences({"seq5": frames})
    assert filtered == {}, "Sequence with unsupported camera should be rejected"


def test_filter_car_sequences_custom_thresholds():
    """Test that custom speed thresholds work correctly."""
    # Create frames with speeds around 36 km/h
    # 0.00009 deg ≈ 14m at equator, over 1.4 seconds ≈ 10 m/s ≈ 36 km/h
    frames = [
        build_frame("f0", "seq6", 0.0, 0.0, 0),
        build_frame("f1", "seq6", 0.00009, 0.00009, 1400),  # ~36 km/h
        build_frame("f2", "seq6", 0.00018, 0.00018, 2800),  # ~36 km/h
    ]
    seqs = {"seq6": frames}

    # Should pass with lower threshold (30-120)
    filtered = filter_car_sequences(seqs, min_speed_kmh=30.0, max_speed_kmh=120.0)
    assert "seq6" in filtered, "Sequence should pass with 30-120 km/h threshold"

    # Should fail with higher threshold (50-120) since max speed is ~36 km/h
    filtered = filter_car_sequences(seqs, min_speed_kmh=50.0, max_speed_kmh=120.0)
    assert "seq6" not in filtered, "Sequence should fail with 50-120 km/h threshold"


def test_filter_car_sequences_mixed_speeds():
    """Test sequence with mixed speeds - should judge by max speed."""
    frames = [
        build_frame("f0", "seq7", 0.0, 0.0, 0),
        build_frame("f1", "seq7", 0.00001, 0.00001, 1000),  # ~4 km/h (stopped)
        build_frame("f2", "seq7", 0.00002, 0.00002, 2000),  # ~4 km/h
        build_frame("f3", "seq7", 0.00017, 0.00017, 3000),  # ~65 km/h (accelerated!)
        build_frame("f4", "seq7", 0.00032, 0.00032, 4000),  # ~65 km/h
    ]
    seqs = {"seq7": frames}

    filtered = filter_car_sequences(seqs, min_speed_kmh=40.0, max_speed_kmh=120.0)
    assert (
        "seq7" in filtered
    ), "Sequence should pass because max speed reaches car range"
