from pathlib import Path
import re

p = Path("tests/test_sfm_colmap_integration.py")
content = p.read_text()

# Remove _build_frames function
content = re.sub(r'def _build_frames.*?return \[.*?\]\s+for i in range\(n_frames\)\s+\]', '', content, flags=re.DOTALL)

# Add sample_loader import
if "get_sample_frames" not in content:
    content = content.replace("from dtm_from_mapillary.geom.sfm_colmap import (", "from tests.sample_loader import get_sample_frames\nfrom dtm_from_mapillary.geom.sfm_colmap import (")

# Fix tests
content = re.sub(r'def test_colmap_run_without_refinement_fails\(\).*?def test_colmap_run_with_full_refinement_fails\(\)', 
'''def test_colmap_run_without_refinement_fails():
    seqs, imagery_root = get_sample_frames()
    from dtm_from_mapillary.geom.colmap_adapter import COLMAPUnavailable
    with pytest.raises(COLMAPUnavailable):
        run(seqs, imagery_root=imagery_root)

def test_colmap_run_with_full_refinement_fails()''', content, flags=re.DOTALL)

content = re.sub(r'def test_colmap_run_with_full_refinement_fails\(\).*?def test_colmap_run_with_quick_refinement_fails\(\)', 
'''def test_colmap_run_with_full_refinement_fails():
    seqs, imagery_root = get_sample_frames()
    from dtm_from_mapillary.geom.colmap_adapter import COLMAPUnavailable
    with pytest.raises(COLMAPUnavailable):
        run(seqs, imagery_root=imagery_root, refine_cameras=True, refinement_method="full")

def test_colmap_run_with_quick_refinement_fails()''', content, flags=re.DOTALL)

content = re.sub(r'def test_colmap_run_with_quick_refinement_fails\(\).*?def test_colmap_refinement_insufficient_points\(\)', 
'''def test_colmap_run_with_quick_refinement_fails():
    seqs, imagery_root = get_sample_frames()
    from dtm_from_mapillary.geom.colmap_adapter import COLMAPUnavailable
    with pytest.raises(COLMAPUnavailable):
        run(seqs, imagery_root=imagery_root, refine_cameras=True, refinement_method="quick")

def test_colmap_refinement_insufficient_points()''', content, flags=re.DOTALL)

content = re.sub(r'def test_colmap_refinement_insufficient_points\(\).*?def test_extract_correspondences_for_frame', 
'''def test_colmap_refinement_insufficient_points():
    seqs, imagery_root = get_sample_frames()
    from dtm_from_mapillary.geom.colmap_adapter import COLMAPUnavailable
    with pytest.raises(COLMAPUnavailable):
        run(seqs, imagery_root=imagery_root, refine_cameras=True)

def test_extract_correspondences_for_frame''', content, flags=re.DOTALL)

content = re.sub(r'frames = _build_frames\("seq1", n_frames=(\d+)\)', r'seqs, _ = get_sample_frames()\n    frames = list(seqs.values())[0][:\1]', content)

content = re.sub(r'seqs = {"seqA": _build_frames\("seqA", n_frames=2\)}', r'seqs, _ = get_sample_frames()\n    seq_id = list(seqs.keys())[0]\n    seqs = {seq_id: seqs[seq_id][:2]}', content)

# Fix test_colmap_fixture_loader
content = re.sub(r'seqs = {"seqA": _build_frames\("seqA", n_frames=2\)}', r'seqs, _ = get_sample_frames()\n    seq_id = list(seqs.keys())[0]\n    seqs = {seq_id: seqs[seq_id][:2]}', content)

p.write_text(content)
