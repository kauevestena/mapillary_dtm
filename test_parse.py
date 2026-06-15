import json
from pathlib import Path

path = Path("out_eval_prod/cache/opensfm/8vebekigjgf13bnbztzfji/reconstruction.json")
payload = json.loads(path.read_text(encoding="utf8"))
shots = payload[0].get("shots", {})

for image_filename in list(shots.keys())[:5]:
    image_id = image_filename.rsplit('.', 1)[0] if '.' in image_filename else image_filename
    print(f"Filename: {image_filename}, ID: {image_id}")
