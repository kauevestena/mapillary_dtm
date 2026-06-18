from pathlib import Path
import re

p = Path("tests/test_monodepth.py")
content = p.read_text()

content = re.sub(r'def test_monodepth_adapter_outputs.*?frames = {"seqA": _build_frames\("seqA", n_frames=2\)}',
'''def test_monodepth_adapter_outputs(tmp_path: Path):
    seqs, _ = get_sample_frames()
    seq_id = list(seqs.keys())[0]
    frames = {seq_id: seqs[seq_id][:2]}''', content, flags=re.DOTALL)

content = content.replace('assert adapter.calls == ["seqA-frame-0", "seqA-frame-1"]',
'''frame_ids = [f.image_id for f in frames[seq_id]]
    assert adapter.calls == frame_ids''')

content = content.replace('seq_res = results["seqA"]', 'seq_res = results[seq_id]')
content = content.replace('depth = seq_res["seqA-frame-0"]["depth"]', 'depth = seq_res[frame_ids[0]]["depth"]')
content = content.replace('uncert = seq_res["seqA-frame-0"]["uncertainty"]', 'uncert = seq_res[frame_ids[0]]["uncertainty"]')

content = re.sub(r'def test_monodepth_adapter_fails_without_model.*?frames = {"seqA": _build_frames\("seqA", n_frames=1\)}',
'''def test_monodepth_adapter_fails_without_model(tmp_path: Path):
    seqs, _ = get_sample_frames()
    seq_id = list(seqs.keys())[0]
    frames = {seq_id: seqs[seq_id][:1]}''', content, flags=re.DOTALL)

p.write_text(content)
