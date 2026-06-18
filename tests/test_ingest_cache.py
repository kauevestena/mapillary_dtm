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


import time
from pathlib import Path
from typing import Dict, List, Sequence

from dtm_from_mapillary.common_core import FrameMeta
from dtm_from_mapillary.ingest.sequence_scan import discover_sequences
from dtm_from_mapillary.ingest.imagery_cache import prefetch_imagery
from dtm_from_mapillary.ingest import cache_utils
from tests.sample_loader import get_sample_frames
import json
SAMPLE_META_PATH = Path(__file__).resolve().parents[1] / "qa" / "data" / "sample_dataset" / "metadata.json"


class _SequenceClient:
    def __init__(self, fail_on_fetch: bool = False):
        self.fail_on_fetch = fail_on_fetch
        self.image_calls = 0
        self.meta_calls = 0
        self.data = json.loads(SAMPLE_META_PATH.read_text())
        self.seq_id = list(self.data.keys())[0]

    def list_sequence_ids_in_bbox(self, bbox: Sequence[float]) -> List[str]:
        return [self.seq_id]

    def list_image_ids_in_sequence(self, seq_id: str, limit: int = 10_000) -> List[str]:
        if self.fail_on_fetch:
            raise AssertionError("list_image_ids_in_sequence should not be invoked when cache present")
        self.image_calls += 1
        return [d["image_id"] for d in self.data[self.seq_id]]

    def get_images_meta(
        self,
        image_ids: Sequence[str],
        fields: Sequence[str] | None = None,
        chunk_size: int = 50,
    ) -> List[Dict]:
        if self.fail_on_fetch:
            raise AssertionError("get_images_meta should not be invoked when cache present")
        self.meta_calls += 1
        records = []
        for img_id in image_ids:
            for d in self.data[self.seq_id]:
                if d["image_id"] == img_id:
                    # Construct raw API format
                    records.append({
                        "id": img_id,
                        "sequence_id": self.seq_id,
                        "geometry": {"coordinates": [d["lon"], d["lat"], 3.0]},
                        "captured_at": d["captured_at_ms"],
                        "camera_type": d["camera_type"],
                        "camera_parameters": d.get("camera_parameters", []),
                        "quality_score": d.get("quality_score", 0.9),
                        "thumb_1024_url": d.get("thumbnail_url"),
                    })
        return records

    def list_images_in_bbox(self, bbox: Sequence[float], limit: int = 2_000) -> List[Dict]:
        return []


class _ImageryClient:
    def __init__(self):
        self.downloaded: Dict[Path, int] = {}

    def download_file(self, url: str, dest_path: Path, chunk_size: int = 1 << 20) -> None:
        dest_path.write_bytes(b"data")
        self.downloaded[dest_path] = self.downloaded.get(dest_path, 0) + 1

    def get_thumbnail_url(self, image_id: str, resolution: int = 1024) -> str:
        return f"https://example.com/{image_id}_{resolution}.jpg"


def _bbox() -> Sequence[float]:
    return (-49.0, -28.0, -48.0, -27.0)


def test_discover_sequences_caches_metadata(tmp_path: Path) -> None:
    client = _SequenceClient()
    cache_dir = tmp_path / "metadata"
    sequences = discover_sequences(
        _bbox(),
        client=client,
        cache_dir=cache_dir,
        use_cache=True,
        force_refresh=False,
    )

    assert client.seq_id in sequences
    frames = sequences[client.seq_id]
    assert len(frames) == 4
    assert all(isinstance(frame, FrameMeta) for frame in frames)
    assert frames[0].thumbnail_url is not None
    cache_file = cache_dir / f"{client.seq_id}.jsonl"
    assert cache_file.exists()
    assert client.image_calls == 1
    assert client.meta_calls == 1

    cached_client = _SequenceClient(fail_on_fetch=True)
    cached_sequences = discover_sequences(
        _bbox(),
        client=cached_client,
        cache_dir=cache_dir,
        use_cache=True,
        force_refresh=False,
    )
    assert cached_client.seq_id in cached_sequences
    assert cached_client.image_calls == 0
    assert cached_client.meta_calls == 0


def test_prefetch_imagery_writes_files(tmp_path: Path) -> None:
    client = _ImageryClient()
    seqs, _ = get_sample_frames()
    seq_id = list(seqs.keys())[0]
    frames = seqs[seq_id][:1]
    
    stats = prefetch_imagery(
        {seq_id: frames},
        client=client,
        cache_dir=tmp_path,
        max_per_sequence=2,
        resolution=512,
    )

    seq_dir = tmp_path / "imagery" / seq_id
    expected_path = seq_dir / f"{frames[0].image_id}_512.jpg"
    assert expected_path.exists()
    assert stats == {seq_id: 1}
    assert client.downloaded[expected_path] == 1


def test_enforce_quota_prunes_oldest(tmp_path: Path) -> None:
    for idx in range(3):
        file_path = tmp_path / f"file-{idx}.bin"
        file_path.write_bytes(b"x" * 100)
        time.sleep(0.01)

    limit_bytes = 150
    limit_gb = limit_bytes / (1024**3)
    total_after, removed = cache_utils.enforce_quota(tmp_path, limit_gb)
    assert total_after <= limit_bytes
    assert removed, "Expected at least one file to be pruned"
