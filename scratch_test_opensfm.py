import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from tests.sample_loader import get_sample_frames
from dtm_from_mapillary.geom.sfm_opensfm import run
seqs, imagery_root = get_sample_frames()
try:
    results = run(seqs, imagery_root=imagery_root)
    print("SUCCESS!", list(results.keys()))
except Exception as e:
    print("FAILED!", type(e), str(e))
