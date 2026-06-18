from pathlib import Path
import re

p = Path("tests/test_ground_masks.py")
content = p.read_text()

content = content.replace("def build_frame(image_id=\"img\", seq_id=\"seq\", width=4000, height=3000):", "from tests.sample_loader import get_sample_frames\ndef build_frame(image_id=\"img\", seq_id=\"seq\", width=4000, height=3000):")

content = re.sub(r'def build_frame.*?return FrameMeta\(.*?quality_score=0\.9,\s*\)', '', content, flags=re.DOTALL)

content = re.sub(r'def test_prepare_creates_masks\(tmp_path\):\s*frame = build_frame\("img-1"\)\s*out = prepare\(\{"seq": \[frame\]\}, out_dir=tmp_path, backend="constant"\)', 
'''def test_prepare_creates_masks(tmp_path):
    seqs, _ = get_sample_frames()
    seq_id = list(seqs.keys())[0]
    frame = seqs[seq_id][0]
    out = prepare({seq_id: [frame]}, out_dir=tmp_path, backend="constant")''', content)

content = content.replace('mask_path = tmp_path / "img-1.npz"', 'mask_path = tmp_path / f"{frame.image_id}.npz"')
content = content.replace('assert out == {"seq": [mask_path]}', 'assert out == {seq_id: [mask_path]}')

content = re.sub(r'def test_prepare_uses_existing_cache\(tmp_path\):\s*frame = build_frame\("img-cache"\)\s*mask_path = tmp_path / "img-cache.npz"',
'''def test_prepare_uses_existing_cache(tmp_path):
    seqs, _ = get_sample_frames()
    seq_id = list(seqs.keys())[0]
    frame = seqs[seq_id][0]
    mask_path = tmp_path / f"{frame.image_id}.npz"''', content)

content = content.replace('np.savez_compressed(mask_path, prob=existing, image_id="img-cache", seq_id="seq")', 'np.savez_compressed(mask_path, prob=existing, image_id=frame.image_id, seq_id=seq_id)')
content = content.replace('prepare({"seq": [frame]}, out_dir=tmp_path, backend="soft-horizon")', 'prepare({seq_id: [frame]}, out_dir=tmp_path, backend="soft-horizon")')

content = re.sub(r'def test_prepare_force_overwrites\(tmp_path\):\s*frame = build_frame\("img-force"\)\s*mask_path = tmp_path / "img-force.npz"',
'''def test_prepare_force_overwrites(tmp_path):
    seqs, _ = get_sample_frames()
    seq_id = list(seqs.keys())[0]
    frame = seqs[seq_id][0]
    mask_path = tmp_path / f"{frame.image_id}.npz"''', content)

content = content.replace('prepare({"seq": [frame]}, out_dir=tmp_path, backend="constant", force=True)', 'prepare({seq_id: [frame]}, out_dir=tmp_path, backend="constant", force=True)')

p.write_text(content)
