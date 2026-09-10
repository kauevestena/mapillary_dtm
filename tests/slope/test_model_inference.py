"""Opt-in actual inference, never monkeypatched or substituted with fake masks."""
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from slope_from_mapillary.cli import prepare_project
from slope_from_mapillary.reconstruction import load_colmap
from slope_from_mapillary.segmentation import segment_project
from slope_from_mapillary.surfaces import ObservationPolicy, select_surfaces

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/"qa/data/sample_dataset"
COLMAP = DATA/"geometry/colmap/l27kwlcx3fjh7t6w9ccvic/sparse_txt/0"


@unittest.skipUnless(os.getenv("SLOPE_TEST_REAL_MODELS") == "1", "Real model inference is opt-in")
class RealInferenceTests(unittest.TestCase):
    def test_actual_model_masks_feed_actual_track_selector(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)/"project.json"
            prepare_project(COLMAP, "colmap", project, metadata=DATA/"metadata.json")
            result = segment_project(project, DATA/"imagery/l27kwlcx3fjh7t6w9ccvic",
                model_id="nvidia/segformer-b0-finetuned-cityscapes-512-1024")
            model = load_colmap(COLMAP)
            self.assertEqual(result["written_masks"], len(model.shots))
            for path in result["files"]:
                with np.load(path, allow_pickle=False) as mask:
                    self.assertTrue(np.isfinite(mask["confidence"]).all())
                    self.assertGreater(mask["confidence"].max(), 0)
                    self.assertGreater(np.count_nonzero(mask["labels"]), 0)
                    self.assertTrue(str(mask["source"].item()).startswith("model:"))
            config = json.loads(project.read_text())
            cloud = select_surfaces(model, project.parent/"surface_masks", config["surfaces"], ObservationPolicy())
            # The real COLMAP model has only two cameras; production requires
            # three views. Real inference must not manufacture that third view.
            self.assertEqual(len(cloud.xyz), 0)
            self.assertEqual(cloud.stats["available_masks"], len(model.shots))


if __name__ == "__main__":
    unittest.main()
