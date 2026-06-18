from pathlib import Path
import re

p = Path("tests/test_ground_masks.py")
content = """from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if "dtm_from_mapillary" not in sys.modules:
    pkg = types.ModuleType("dtm_from_mapillary")
    pkg.__path__ = [str(ROOT)]
    sys.modules["dtm_from_mapillary"] = pkg
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dtm_from_mapillary.semantics.ground_masks import prepare
from tests.sample_loader import get_sample_frames

def test_prepare_missing_model_fails(tmp_path):
    seqs, _ = get_sample_frames()
    seq_id = list(seqs.keys())[0]
    frame = seqs[seq_id][0]
    
    with pytest.raises(RuntimeError, match="Ground mask missing"):
        prepare({seq_id: [frame]}, out_dir=tmp_path)


def test_prepare_uses_existing_cache(tmp_path):
    seqs, _ = get_sample_frames()
    seq_id = list(seqs.keys())[0]
    frame = seqs[seq_id][0]
    mask_path = tmp_path / f"{frame.image_id}.npz"
    existing = np.full((10, 10), 0.42, dtype=np.float32)
    
    # Save cache with proper provenance
    np.savez_compressed(
        mask_path, 
        prob=existing, 
        image_id=frame.image_id, 
        seq_id=seq_id,
        source_type="model",
        backend="test",
        model_id="test",
        model_revision="test"
    )

    prepare({seq_id: [frame]}, out_dir=tmp_path)

    with np.load(mask_path) as data:
        prob = data["prob"]
        assert prob.shape == (10, 10)
        assert np.isclose(prob.mean(), 0.42)


def test_prepare_force_overwrites_fails_without_model(tmp_path):
    seqs, _ = get_sample_frames()
    seq_id = list(seqs.keys())[0]
    frame = seqs[seq_id][0]
    mask_path = tmp_path / f"{frame.image_id}.npz"
    
    # Save existing cache
    np.savez_compressed(
        mask_path, 
        prob=np.zeros((5, 5), dtype=np.float32),
        image_id=frame.image_id, 
        seq_id=seq_id,
        source_type="model",
        backend="test",
        model_id="test",
        model_revision="test"
    )

    # Force should ignore cache and try to predict, failing because model is absent
    with pytest.raises(RuntimeError, match="Ground mask missing"):
        prepare({seq_id: [frame]}, out_dir=tmp_path, force=True)
"""
p.write_text(content)
