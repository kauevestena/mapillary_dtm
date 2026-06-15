import json
from pathlib import Path
from dtm_from_mapillary.ingest.cache_utils import sequence_imagery_dir
from dtm_from_mapillary.geom.opensfm_adapter import OpenSfMRunner
from dtm_from_mapillary.cli.pipeline import filter_car_sequences
import gzip

# How to get the actual `seqs` dictionary with FrameMeta?
# The `run_state.json` contains `inputs_fingerprint`.
# Let's just find the `frame_meta` from Mapillary cache.
# OpenSfM's `dataset_dir/reconstruction.json` has shots.

path = Path("out_eval_prod/cache/opensfm/8vebekigjgf13bnbztzfji/reconstruction.json")
payload = json.loads(path.read_text(encoding="utf8"))
shots = payload[0].get("shots", {})

print("Shots from reconstruction.json:")
for image_filename in list(shots.keys())[:5]:
    image_id = image_filename.rsplit('.', 1)[0] if '.' in image_filename else image_filename
    print(f"  {image_id}")

print("Let's look at the actual images downloaded:")
images = list(Path("out_eval_prod/cache/opensfm/8vebekigjgf13bnbztzfji/images").glob("*"))
print("Images staged in workspace:")
for img in images[:5]:
    print(f"  {img.name}")
