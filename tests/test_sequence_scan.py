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

import json

from dtm_from_mapillary.ingest.sequence_scan import discover_sequences
from dtm_from_mapillary.common_core import FrameMeta


class DummyClient:
    def __init__(self):
        self.requested_sequences = []

    def list_sequence_ids_in_bbox(self, bbox):
        assert bbox == (-1.0, -1.0, 1.0, 1.0)
        return {"seqA", "seqB"}

    def list_image_ids_in_sequence(self, seq_id):
        self.requested_sequences.append(seq_id)
        if seq_id == "seqA":
            return ["img-1", "img-2"]
        if seq_id == "seqB":
            return ["img-3"]
        return []

    def get_images_meta(self, image_ids):
        payloads = {
            "img-1": {
                "id": "img-1",
                "sequence_id": "seqA",
                "captured_at": 1_700_000_000_000,
                "geometry": {"type": "Point", "coordinates": [0.0, 0.0, 5.0]},
                "camera_type": "perspective",
                "camera_parameters": {"fx": 1000.0, "fy": 1000.0, "cx": 500.0, "cy": 500.0},
                "quality_score": 0.9,
            },
            "img-2": {
                "id": "img-2",
                "sequence_id": "seqA",
                "captured_at": "2023-04-01T12:00:00Z",
                "geometry": {"type": "Point", "coordinates": [0.0, 2.0]},  # outside bbox
                "camera_type": "perspective",
                "camera_parameters": '{}',
                "quality_score": 0.8,
            },
            "img-3": {
                "id": "img-3",
                "sequence_id": "seqC",  # wrong sequence, should be discarded
                "captured_at": 1_700_000_001_000,
                "geometry": {"type": "Point", "coordinates": [0.1, 0.1]},
                "camera_type": "perspective",
                "camera_parameters": {"fx": 900.0, "fy": 900.0, "cx": 450.0, "cy": 450.0},
                "quality_score": 0.7,
            },
        }
        return [payloads[i] for i in image_ids]


def test_discover_sequences_filters_bbox_and_sequence(tmp_path):
    client = DummyClient()
    bbox = (-1.0, -1.0, 1.0, 1.0)
    result = discover_sequences(bbox, client=client, cache_dir=tmp_path)

    assert set(result.keys()) == {"seqA"}
    frames = result["seqA"]
    assert len(frames) == 1

    frame = frames[0]
    assert frame.image_id == "img-1"
    assert frame.seq_id == "seqA"
    assert frame.lon == 0.0 and frame.lat == 0.0
    assert frame.alt_ellip == 5.0
    assert frame.camera_type == "perspective"
    assert frame.quality_score == 0.9
    assert frame.cam_params["fx"] == 1000.0


def test_discover_sequences_uses_cache(tmp_path):
    # Prime cache
    seq_id = "cache-seq"
    cache_file = tmp_path / f"{seq_id}.jsonl"
    frame = FrameMeta(
        image_id="img-10",
        seq_id=seq_id,
        captured_at_ms=5,
        lon=0.0,
        lat=0.0,
        alt_ellip=None,
        camera_type="perspective",
        cam_params={},
        quality_score=0.5,
    )
    cache_file.write_text(
        "\n".join(
            [
                "",  # empty line should be ignored
                "{\"bad\": true}",
                json.dumps(frame.to_dict()),
            ]
        ),
        encoding="utf8",
    )

    class CacheClient(DummyClient):
        def list_sequence_ids_in_bbox(self, bbox):
            return {seq_id}

    # Using cache via helper so we don't trigger actual API usage
    result = discover_sequences(
        (-1.0, -1.0, 1.0, 1.0),
        client=CacheClient(),
        cache_dir=tmp_path,
        use_cache=True,
    )
    assert seq_id in result
    assert len(result[seq_id]) == 1
