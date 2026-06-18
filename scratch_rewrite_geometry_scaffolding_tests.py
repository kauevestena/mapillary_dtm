from pathlib import Path
import re

p = Path("tests/test_geometry_scaffolding.py")
content = p.read_text()

# Add sample_loader import
if "get_sample_frames" not in content:
    content = content.replace("from dtm_from_mapillary.geom.vo_simplified import run as run_vo", "from tests.sample_loader import get_sample_frames\nfrom dtm_from_mapillary.geom.vo_simplified import run as run_vo")

# Replace mock frames in test_scaffold_returns_dict
content = re.sub(r'def test_scaffold_returns_dict\(\).*?def test_scaffold_passes_through_options\(\)',
'''def test_scaffold_returns_dict():
    seqs, _ = get_sample_frames()
    results = scaffold_sequence_geometry(seqs)
    
    seq_id = list(seqs.keys())[0]
    assert seq_id in results
    assert "source" in results[seq_id].metadata
    assert "reconstruction" in results[seq_id].metadata

def test_scaffold_passes_through_options()''', content, flags=re.DOTALL)

# Replace mock frames in test_scaffold_passes_through_options
content = re.sub(r'def test_scaffold_passes_through_options\(\).*?def test_fallback_logic_when_no_fixtures\(\)',
'''def test_scaffold_passes_through_options():
    seqs, imagery_root = get_sample_frames()
    results = scaffold_sequence_geometry(seqs, refine_cameras=True)
    seq_id = list(seqs.keys())[0]
    
    assert results[seq_id].metadata.get("cameras_refined") is True or "fixture" in results[seq_id].metadata

def test_fallback_logic_when_no_fixtures()''', content, flags=re.DOTALL)

# Wait, I should rewrite the whole file because there are a few tests. Let's just do that.
