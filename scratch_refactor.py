import re

with open('cli/pipeline.py', 'r') as f:
    content = f.read()

# 1. Remove args from run_pipeline signature
content = re.sub(r'\s*allow_synthetic:\s*bool\s*=\s*False,', '', content)
content = re.sub(r'\s*strict_production:\s*bool\s*=\s*True,', '', content)
content = re.sub(r'\s*vo_force_synthetic:\s*bool\s*=\s*False,', '', content)

# 2. Remove docstrings
content = re.sub(r'\s*allow_synthetic\s*:\s*bool.*?(?=\n\s*[a-z_]+\s*:)', '', content, flags=re.DOTALL)
content = re.sub(r'\s*strict_production\s*:\s*bool.*?(?=\n\s*[a-z_]+\s*:)', '', content, flags=re.DOTALL)
content = re.sub(r'\s*vo_force_synthetic\s*:\s*bool.*?(?=\n\s*[a-z_]+\s*:)', '', content, flags=re.DOTALL)

# 3. Remove validation blocks
validation_block = """    if strict_production and allow_synthetic:
        raise ValueError("--allow-synthetic cannot be combined with strict production mode")"""
content = content.replace(validation_block, '')

# 4. In _run_inputs_fingerprint calls
content = re.sub(r'\s*strict_production=strict_production,', '', content)
content = re.sub(r'\s*allow_synthetic=allow_synthetic,', '', content)

# 5. strict_production preflight is now mandatory
content = re.sub(r'    if strict_production:\n(.*?)    else:\n        run_state\.complete\(\n            "preflight",\n            counts=\{"strict_production": False\},\n            warnings=\["strict production preflight skipped"\],\n        \)', r'\1', content, flags=re.DOTALL)

# 6. Remove allow_synthetic and vo_force_synthetic passes
content = re.sub(r'\s*allow_synthetic=allow_synthetic,', '', content)
content = re.sub(r'\s*allow_heuristic=allow_synthetic,', '', content)
content = re.sub(r'\s*allow_synthetic_depth=allow_synthetic,', '', content)
content = re.sub(r'\s*vo_force_synthetic=vo_force_synthetic,', '', content)
content = re.sub(r'\s*force_synthetic=vo_force_synthetic,', '', content)
content = re.sub(r'\s*strict=not allow_synthetic', 'strict=True', content)
content = re.sub(r'\s*include_plane_sweep=allow_synthetic if name != "vo" else False,', 'include_plane_sweep=False,', content)

# 7. Remove fallback conditional blocks
colmap_fallback = """        if not allow_synthetic and not reconB:
            from ..geom.colmap_adapter import COLMAPUnavailable
            raise COLMAPUnavailable("COLMAP synthetic fallback disabled and no real reconstruction was produced")"""
colmap_new = """        if not reconB:
            from ..geom.colmap_adapter import COLMAPUnavailable
            raise COLMAPUnavailable("COLMAP no real reconstruction was produced")"""
content = content.replace(colmap_fallback, colmap_new)

# 8. Remove from run_pipeline arguments
content = re.sub(r'\s*"allow_synthetic":\s*allow_synthetic,', '', content)
content = re.sub(r'\s*"allow_heuristic":\s*allow_synthetic,', '', content)
content = re.sub(r'\s*"strict_production":\s*strict_production,', '', content)
content = re.sub(r'\s*"force_synthetic":\s*vo_force_synthetic,', '', content)

# 9. Typer args
content = re.sub(r'\s*allow_synthetic:\s*bool\s*=\s*typer\.Option\([^)]*\),', '', content, flags=re.DOTALL)
content = re.sub(r'\s*strict_production:\s*bool\s*=\s*typer\.Option\([^)]*\),', '', content, flags=re.DOTALL)
content = re.sub(r'\s*vo_force_synthetic:\s*bool\s*=\s*typer\.Option\([^)]*\),', '', content, flags=re.DOTALL)

# 10. `_run_inputs_fingerprint` signature
content = re.sub(r'\s*allow_synthetic:\s*bool,', '', content)
content = re.sub(r'\s*strict_production:\s*bool,', '', content)
content = re.sub(r'\s*"allow_synthetic":\s*allow_synthetic,', '', content)
content = re.sub(r'\s*"strict_production":\s*strict_production,', '', content)

# 11. `_strict_preflight` signature
content = re.sub(r'\s*vo_force_synthetic:\s*bool,', '', content)
content = re.sub(r'\s*if not forced_envs and not vo_force_synthetic\n\s*else f"forced synthetic flags present: \{forced_envs\}, vo_force_synthetic=\{vo_force_synthetic\}",', 'if not forced_envs else f"forced synthetic flags present: {forced_envs}",', content, flags=re.DOTALL)
content = re.sub(r'\s*not forced_envs and not vo_force_synthetic,', 'not forced_envs,', content)


with open('cli/pipeline.py', 'w') as f:
    f.write(content)

print("Refactored pipeline.py")
