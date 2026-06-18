from pathlib import Path
import re

p = Path("tests/test_monodepth.py")
content = p.read_text()

# Add sample_loader import
if "get_sample_frames" not in content:
    content = content.replace("from dtm_from_mapillary.geom.monodepth import run", "from tests.sample_loader import get_sample_frames\nfrom dtm_from_mapillary.geom.monodepth import run")

# Replace mock frames with get_sample_frames in test_monodepth_fails_without_imagery
content = re.sub(r'def test_monodepth_fails_without_imagery\(\).*?def test_monodepth_dummy_model\(\)',
'''def test_monodepth_fails_without_imagery():
    seqs, imagery_root = get_sample_frames()
    with pytest.raises(RuntimeError, match="Monodepth inference failed"):
        # We pass a fake directory so it fails
        run(seqs, imagery_root=Path("/tmp/nonexistent"))

def test_monodepth_dummy_model()''', content, flags=re.DOTALL)

# In test_monodepth_dummy_model, use real imagery_root to succeed?
# Wait, test_monodepth_dummy_model expects dummy model or what? Let's check what it does.
