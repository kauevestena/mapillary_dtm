import re

def rewrite_test(filepath, replacements):
    with open(filepath, 'r') as f:
        content = f.read()
    
    for old, new in replacements:
        content = content.replace(old, new)
        
    with open(filepath, 'w') as f:
        f.write(content)
        
rewrite_test("tests/test_ground_masks.py", [
    ("""def test_prepare_creates_masks(tmp_path: Path):
    frame = _frame()
    results = ground_masks.prepare({"seq-1": [frame]}, out_dir=tmp_path)
    assert "seq-1" in results
    assert "prob" in results["seq-1"]["img-1"]
    assert results["seq-1"]["img-1"]["prob"].shape == (96, 160)
    assert (tmp_path / "img-1.npz").exists()""", """def test_prepare_creates_masks_fails_without_model(tmp_path: Path):
    frame = _frame()
    with pytest.raises(RuntimeError, match="Ground mask missing"):
        ground_masks.prepare({"seq-1": [frame]}, out_dir=tmp_path)"""),
        
    ("""def test_prepare_uses_existing_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    frame = _frame()
    # Create fake cache
    np.savez_compressed(tmp_path / "img-1.npz", prob=np.ones((4, 4), dtype=np.float32))

    # Disable the model to ensure it's not used
    monkeypatch.setattr(ground_masks, "_should_init_model_masker", lambda **kwargs: False)

    results = ground_masks.prepare({"seq-1": [frame]}, out_dir=tmp_path)

    assert results["seq-1"]["img-1"]["prob"].mean() == 1.0""", """def test_prepare_uses_existing_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    frame = _frame()
    # Create fake cache with provenance
    np.savez_compressed(
        tmp_path / "img-1.npz", 
        prob=np.ones((4, 4), dtype=np.float32),
        source_type="model",
        backend="test"
    )

    results = ground_masks.prepare({"seq-1": [frame]}, out_dir=tmp_path)

    assert results["seq-1"]["img-1"]["prob"].mean() == 1.0"""),
    
    ("""def test_prepare_force_overwrites(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    frame = _frame()
    np.savez_compressed(tmp_path / "img-1.npz", prob=np.ones((4, 4), dtype=np.float32))

    monkeypatch.setattr(ground_masks, "_should_init_model_masker", lambda **kwargs: False)

    results = ground_masks.prepare({"seq-1": [frame]}, out_dir=tmp_path, force=True)
    # The heuristic mask is ~0.5 mean, not 1.0
    assert results["seq-1"]["img-1"]["prob"].mean() < 0.9""", """def test_prepare_force_overwrites_fails_without_model(tmp_path: Path):
    frame = _frame()
    np.savez_compressed(
        tmp_path / "img-1.npz", 
        prob=np.ones((4, 4), dtype=np.float32),
        source_type="model",
        backend="test"
    )
    # With force=True, cache is ignored, so it needs the model and fails
    with pytest.raises(RuntimeError, match="Ground mask missing"):
        ground_masks.prepare({"seq-1": [frame]}, out_dir=tmp_path, force=True)""")
])

