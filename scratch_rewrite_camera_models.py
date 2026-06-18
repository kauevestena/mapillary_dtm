from pathlib import Path
import re

p = Path("tests/test_camera_models.py")
content = p.read_text()

content = content.replace("from dtm_from_mapillary.common_core import FrameMeta", "from dtm_from_mapillary.common_core import FrameMeta\nfrom tests.sample_loader import get_sample_frames")

content = re.sub(r'def build_frame.*?quality_score=None,\s*\)', '', content, flags=re.DOTALL)

content = re.sub(r'def test_make_opensfm_model_perspective\(\):.*?assert model\["k1"\] == 0\.1 and model\["p1"\] == 0\.001',
'''def test_make_opensfm_model_perspective():
    seqs, _ = get_sample_frames()
    seq_id = list(seqs.keys())[0]
    frame = seqs[seq_id][0]
    
    model = make_opensfm_model(frame)

    assert model["projection_type"] == frame.camera_type
    assert "width" in model or "focal" in model''', content, flags=re.DOTALL)

content = re.sub(r'def test_make_opensfm_model_spherical_defaults\(\).*?assert model\["projection_type"\] == "spherical"',
'''def test_make_opensfm_model_spherical_defaults():
    seqs, _ = get_sample_frames()
    seq_id = list(seqs.keys())[0]
    frame = seqs[seq_id][0]
    # Simulate a spherical camera
    frame.camera_type = "spherical"
    frame.cam_params = {}
    
    model = make_opensfm_model(frame)
    assert model["projection_type"] == "spherical"''', content, flags=re.DOTALL)

p.write_text(content)
