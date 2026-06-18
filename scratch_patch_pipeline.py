from pathlib import Path

p = Path("cli/pipeline.py")
content = p.read_text()

target = 'timings["masks_s"] = time.perf_counter() - t0'
insertion = """
    filter_sequences_by_mask(seqs, mask_cache_dir)
    if not seqs:
        log.warning("All sequences dropped due to insufficient road masks. Stopping pipeline early.")
        return json.loads((out_path / "manifest.json").read_text(encoding="utf8")) if (out_path / "manifest.json").exists() else {}
"""

content = content.replace(target, target + "\n" + insertion)
p.write_text(content)
