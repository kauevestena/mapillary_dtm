import os

def rewrite_file(filepath, replacements):
    with open(filepath, 'r') as f:
        content = f.read()
    
    for old, new in replacements:
        if old not in content:
            print(f"FAILED to find in {filepath}:\n{old}")
            return False
        content = content.replace(old, new)
        
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Successfully rewritten {filepath}")
    return True

# monodepth.py
rewrite_file("depth/monodepth.py", [
    ("    allow_synthetic: bool = True,\n", ""),
    ("""                if _should_init_model_mono(model_path, allow_synthetic=allow_synthetic):
                    from ..ml.monodepth_adapter import MiDaSAdapter

                    model = MiDaSAdapter(
                        model_path=model_path,
                        device=os.environ.get("MONODEPTH_DEVICE", "cpu"),
                    )
                else:
                    model = None""", """                from ..ml.monodepth_adapter import MiDaSAdapter

                model = MiDaSAdapter(
                    model_path=model_path,
                    device=os.environ.get("MONODEPTH_DEVICE", "cpu"),
                )"""),
    ("""                if model is None:
                    if not allow_synthetic:
                        raise RuntimeError(
                            "Monodepth missing and synthetic disabled. "
                            "Set MONODEPTH_MODEL_PATH to a valid torchscript model."
                        )
                    dmap = _synthetic_depth_map(w, h, rng)
                    # use slightly less reliable quality for synthetic
                    metadata["quality"] = 0.6
                else:""", """                if model is None:
                    raise RuntimeError("Monodepth model failed to initialize.")
                else:"""),
])

# ground_masks.py
rewrite_file("semantics/ground_masks.py", [
    ("    allow_heuristic: bool = True,\n", ""),
    ("require_provenance = not allow_heuristic", "require_provenance = True"),
    ("""                if not model_initialized:
                    if _should_init_model_masker(model_path, allow_heuristic=allow_heuristic):
                        model = _init_model_masker(model_path=model_path, imagery_root=imagery_root)
                    model_initialized = True""", """                if not model_initialized:
                    model = _init_model_masker(model_path=model_path, imagery_root=imagery_root)
                    model_initialized = True"""),
    ("""                if prob is None:
                    if not allow_heuristic:
                        raise RuntimeError(
                            "Ground mask missing and heuristic masks are disabled. "
                            "Provide provenanced cached masks, set GROUND_MASK_MODEL_PATH, "
                            "or cache the configured Hugging Face ground model."
                        )
                    prob = _synthesize_mask(frame, backend=backend)
                    provenance = {
                        "source_type": "heuristic",
                        "backend": backend,
                        "model_id": None,
                        "model_revision": None,
                    }""", """                if prob is None:
                    raise RuntimeError(
                        "Ground mask missing. "
                        "Provide provenanced cached masks, set GROUND_MASK_MODEL_PATH, "
                        "or cache the configured Hugging Face ground model."
                    )""")
])

