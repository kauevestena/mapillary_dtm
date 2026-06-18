import re

def rewrite_test(filepath, replacements):
    with open(filepath, 'r') as f:
        content = f.read()
    
    for old, new in replacements:
        if old not in content:
            print(f"FAILED: {old[:50]}...")
            return False
        content = content.replace(old, new)
        
    with open(filepath, 'w') as f:
        f.write(content)
        
rewrite_test("tests/test_monodepth.py", [
    ("""def test_monodepth_adapter_fallback(tmp_path: Path):
    frame = _frame()
    results = predict_depths({"seq-1": [frame]}, out_dir=tmp_path)
    # The default mock returns synthetic flat depth maps
    dmap = results["seq-1"]["img-1"]["depth"]
    assert np.all(dmap >= 3.0)""", """def test_monodepth_adapter_fails_without_model(tmp_path: Path):
    frame = _frame()
    with pytest.raises(RuntimeError, match="Monodepth prediction unavailable"):
        predict_depths({"seq-1": [frame]}, out_dir=tmp_path)""")
])

