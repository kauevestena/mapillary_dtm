from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Sequence

from common_core import FrameMeta
from ingest.sequence_scan import discover_sequences
from ingest.imagery_cache import prefetch_imagery
from ingest import cache_utils


class _SequenceClient:
    def __init__(self, fail_on_fetch: bool = False):
        self.fail_on_fetch = fail_on_fetch
        self.image_calls = 0
        self.meta_calls = 0

    def list_sequence_ids_in_bbox(self, bbox: Sequence[float]) -> List[str]:
        return ["seq-1"]

    def list_image_ids_in_sequence(self, seq_id: str, limit: int = 10_000) -> List[str]:
        if self.fail_on_fetch:
            raise AssertionError("list_image_ids_in_sequence should not be invoked when cache present")
        self.image_calls += 1
        return ["img-1", "img-2"]

    def get_images_meta(
        self,
        image_ids: Sequence[str],
        fields: Sequence[str] | None = None,
        chunk_size: int = 50,
    ) -> List[Dict]:
        if self.fail_on_fetch:
            raise AssertionError("get_images_meta should not be invoked when cache present")
        self.meta_calls += 1
        records: List[Dict] = []
        for idx, image_id in enumerate(image_ids):
            records.append(
                {
                    "id": image_id,
                    "sequence_id": "seq-1",
                    "geometry": {"coordinates": [1.0 + idx, 2.0 + idx, 3.0]},
                    "captured_at": 0,
                    "camera_type": "perspective",
                    "camera_parameters": {},
                    "quality_score": 0.9,
                    "thumb_1024_url": f"https://example.com/{image_id}.jpg",
                }
            )
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
    return (-1.0, -1.0, 1.0, 1.0)


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

    assert "seq-1" in sequences
    frames = sequences["seq-1"]
    assert len(frames) == 2
    assert all(isinstance(frame, FrameMeta) for frame in frames)
    assert frames[0].thumbnail_url == "https://example.com/img-1.jpg"
    cache_file = cache_dir / "seq-1.jsonl"
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
    assert "seq-1" in cached_sequences
    assert cached_client.image_calls == 0
    assert cached_client.meta_calls == 0


def test_prefetch_imagery_writes_files(tmp_path: Path) -> None:
    client = _ImageryClient()
    frames = [
        FrameMeta(
            image_id="img-10",
            seq_id="seq-10",
            captured_at_ms=0,
            lon=1.0,
            lat=1.0,
            alt_ellip=0.0,
            camera_type="perspective",
            cam_params={},
            quality_score=0.9,
            thumbnail_url="https://example.com/img-10.jpg",
        )
    ]
    stats = prefetch_imagery(
        {"seq-10": frames},
        client=client,
        cache_dir=tmp_path,
        max_per_sequence=2,
        resolution=512,
    )

    seq_dir = tmp_path / "imagery" / "seq-10"
    expected_path = seq_dir / "img-10_512.jpg"
    assert expected_path.exists()
    assert stats == {"seq-10": 1}
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
