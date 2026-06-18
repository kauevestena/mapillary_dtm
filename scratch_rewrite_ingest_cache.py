import json
from pathlib import Path
import re

p = Path("tests/test_ingest_cache.py")
content = p.read_text()

content = content.replace("from dtm_from_mapillary.ingest import cache_utils", "from dtm_from_mapillary.ingest import cache_utils\nfrom tests.sample_loader import get_sample_frames\nimport json\nSAMPLE_META_PATH = Path(__file__).resolve().parents[1] / \"qa\" / \"data\" / \"sample_dataset\" / \"metadata.json\"")

content = re.sub(r'class _SequenceClient:.*?def list_images_in_bbox\(self, bbox: Sequence\[float\], limit: int = 2_000\) -> List\[Dict\]:\s*return \[\]',
'''class _SequenceClient:
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
                if str(d["id"]) == img_id or d["image_id"] == img_id:
                    records.append(d)
        return records

    def list_images_in_bbox(self, bbox: Sequence[float], limit: int = 2_000) -> List[Dict]:
        return []''', content, flags=re.DOTALL)

content = content.replace('assert "seq-1" in sequences\n    frames = sequences["seq-1"]', 'assert client.seq_id in sequences\n    frames = sequences[client.seq_id]')
content = content.replace('assert len(frames) == 2', 'assert len(frames) == 4')
content = content.replace('assert frames[0].thumbnail_url == "https://example.com/img-1.jpg"', 'assert frames[0].thumbnail_url is not None')
content = content.replace('cache_file = cache_dir / "seq-1.jsonl"', 'cache_file = cache_dir / f"{client.seq_id}.jsonl"')
content = content.replace('assert "seq-1" in cached_sequences', 'assert cached_client.seq_id in cached_sequences')

content = re.sub(r'def test_prefetch_imagery_writes_files.*?stats == {"seq-10": 1}\n    assert client.downloaded\[expected_path\] == 1',
'''def test_prefetch_imagery_writes_files(tmp_path: Path) -> None:
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
    assert client.downloaded[expected_path] == 1''', content, flags=re.DOTALL)

p.write_text(content)
