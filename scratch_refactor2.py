import re

with open('cli/pipeline.py', 'r') as f:
    content = f.read()

def replace_exact(old, new, content):
    if old not in content:
        raise ValueError(f"Could not find exact text:\n{old}")
    return content.replace(old, new)

# 1. run_pipeline signature
content = replace_exact("""    reference_nodata_values: Optional[str] = None,
    allow_synthetic: bool = False,
    strict_production: bool = True,""",
"""    reference_nodata_values: Optional[str] = None,""", content)

content = replace_exact("""    colmap_use_gpu: bool = constants.COLMAP_USE_GPU,
    legacy_vo: bool = False,
    vo_force_synthetic: bool = False,
    vo_min_inliers: Optional[int] = None,""",
"""    colmap_use_gpu: bool = constants.COLMAP_USE_GPU,
    legacy_vo: bool = False,
    vo_min_inliers: Optional[int] = None,""", content)

# 2. _run_inputs_fingerprint calls
content = replace_exact("""        dataset_dir=dataset_path,
        imagery_root=imagery_root_path,
        reference_dtm=reference_dtm,
        strict_production=strict_production,
        allow_synthetic=allow_synthetic,
    )""",
"""        dataset_dir=dataset_path,
        imagery_root=imagery_root_path,
        reference_dtm=reference_dtm,
    )""", content)

# 3. preflight block
content = replace_exact("""    if strict_production:
        preflight = _run_stage(
            run_state,
            "preflight",
            lambda: _strict_preflight(
                seqs,
                imagery_root_path=imagery_root_path,
                reference_dtm=reference_dtm,
                vo_force_synthetic=vo_force_synthetic,
            ),
            inputs={"strict_production": strict_production},
        )
    else:
        run_state.complete(
            "preflight",
            counts={"strict_production": False},
            warnings=["strict production preflight skipped"],
        )""",
"""    preflight = _run_stage(
        run_state,
        "preflight",
        lambda: _strict_preflight(
            seqs,
            imagery_root_path=imagery_root_path,
            reference_dtm=reference_dtm,
        ),
        inputs={"strict_production": True},
    )""", content)

# 4. _strict_preflight definition
content = replace_exact("""def _strict_preflight(
    seqs: dict[str, list[FrameMeta]],
    *,
    imagery_root_path: Optional[Path] = None,
    reference_dtm: Optional[str] = None,
    vo_force_synthetic: bool = False,
) -> dict[str, Any]:""",
"""def _strict_preflight(
    seqs: dict[str, list[FrameMeta]],
    *,
    imagery_root_path: Optional[Path] = None,
    reference_dtm: Optional[str] = None,
) -> dict[str, Any]:""", content)

content = replace_exact("""        if not forced_envs and not vo_force_synthetic
        else f"forced synthetic flags present: {forced_envs}, vo_force_synthetic={vo_force_synthetic}",""",
"""        if not forced_envs
        else f"forced synthetic flags present: {forced_envs}",""", content)

content = replace_exact("""    if not forced_envs and not vo_force_synthetic:
        log.info("Strict production environment looks good.")""",
"""    if not forced_envs:
        log.info("Strict production environment looks good.")""", content)

# 5. mask_index
content = replace_exact("""        lambda: prepare_masks(
            seqs,
            out_dir=mask_cache_dir,
            imagery_root=imagery_root_path,
            allow_heuristic=allow_synthetic,
            progress=progress_enabled,
        ),
        inputs={
            "cache_dir": str(mask_cache_dir),
            "allow_heuristic": allow_synthetic,
            "resume": resume,
        },
        outputs=lambda result: {"cache_dir": str(mask_cache_dir)},
        counts=lambda result: _mask_cache_counts(seqs, mask_cache_dir, strict=not allow_synthetic),""",
"""        lambda: prepare_masks(
            seqs,
            out_dir=mask_cache_dir,
            imagery_root=imagery_root_path,
            progress=progress_enabled,
        ),
        inputs={
            "cache_dir": str(mask_cache_dir),
            "resume": resume,
        },
        outputs=lambda result: {"cache_dir": str(mask_cache_dir)},
        counts=lambda result: _mask_cache_counts(seqs, mask_cache_dir, strict=True),""", content)

# 6. reconA
content = replace_exact("""        lambda: run_opensfm(
            seqs,
            imagery_root=imagery_root_path,
            workspace_root=out_path / "cache" / "opensfm",
            allow_synthetic=allow_synthetic,
            progress=progress_enabled,
        ),""",
"""        lambda: run_opensfm(
            seqs,
            imagery_root=imagery_root_path,
            workspace_root=out_path / "cache" / "opensfm",
            progress=progress_enabled,
        ),""", content)

# 7. reconB
content = replace_exact("""        if not allow_synthetic and not reconB:
            from ..geom.colmap_adapter import COLMAPUnavailable
            raise COLMAPUnavailable("COLMAP synthetic fallback disabled and no real reconstruction was produced")""",
"""        if not reconB:
            from ..geom.colmap_adapter import COLMAPUnavailable
            raise COLMAPUnavailable("COLMAP no real reconstruction was produced")""", content)

content = replace_exact("""        reconB = _run_stage(
            run_state,
            "colmap",
            lambda: run_colmap(
                seqs,
                threads=colmap_threads,
                use_gpu=colmap_use_gpu,
                workspace_root=out_path / "cache" / "colmap",
                imagery_root=imagery_root_path,
                allow_synthetic=allow_synthetic,
                progress=progress_enabled,
            ),""",
"""        reconB = _run_stage(
            run_state,
            "colmap",
            lambda: run_colmap(
                seqs,
                threads=colmap_threads,
                use_gpu=colmap_use_gpu,
                workspace_root=out_path / "cache" / "colmap",
                imagery_root=imagery_root_path,
                progress=progress_enabled,
            ),""", content)

# 8. vo
content = replace_exact("""        vo = _run_stage(
            run_state,
            "vo",
            lambda: run_vo(
                seqs,
                imagery_root=imagery_root_path,
                force_synthetic=vo_force_synthetic,
                min_inliers=vo_min_inliers,
                allow_synthetic=allow_synthetic,
                progress=progress_enabled,
            ),
            inputs={"force_synthetic": vo_force_synthetic, "min_inliers": vo_min_inliers},""",
"""        vo = _run_stage(
            run_state,
            "vo",
            lambda: run_vo(
                seqs,
                imagery_root=imagery_root_path,
                min_inliers=vo_min_inliers,
                progress=progress_enabled,
            ),
            inputs={"min_inliers": vo_min_inliers},""", content)

# 9. anchors
content = replace_exact("""        lambda: (
            lambda found_anchors: (
                found_anchors,
                *solve_scale_and_h(reconA, reconB, vo, found_anchors, seqs),
            )
        )(find_anchors(seqs, token=token, allow_synthetic=allow_synthetic)),""",
"""        lambda: (
            lambda found_anchors: (
                found_anchors,
                *solve_scale_and_h(reconA, reconB, vo, found_anchors, seqs),
            )
        )(find_anchors(seqs, token=token)),""", content)

# 10. mono_depths
content = replace_exact("""        lambda: predict_depths(
            seqs,
            out_dir=depth_cache_dir,
            imagery_root=imagery_root_path,
            allow_synthetic=allow_synthetic,
            progress=progress_enabled,
        ),
        inputs={
            "cache_dir": str(depth_cache_dir),
            "allow_synthetic": allow_synthetic,
            "resume": resume,
        },
        outputs=lambda result: {"cache_dir": str(depth_cache_dir)},
        counts=lambda result: _depth_cache_counts(seqs, depth_cache_dir, strict=not allow_synthetic),""",
"""        lambda: predict_depths(
            seqs,
            out_dir=depth_cache_dir,
            imagery_root=imagery_root_path,
            progress=progress_enabled,
        ),
        inputs={
            "cache_dir": str(depth_cache_dir),
            "resume": resume,
        },
        outputs=lambda result: {"cache_dir": str(depth_cache_dir)},
        counts=lambda result: _depth_cache_counts(seqs, depth_cache_dir, strict=True),""", content)

# 11. ground_extract
content = replace_exact("""                mask_dir=mask_cache_dir,
                mono_cache=depth_cache_dir,
                vo_recon=vo,
                imagery_root=imagery_root_path,
                include_plane_sweep=allow_synthetic if name != "vo" else False,
                allow_synthetic_depth=allow_synthetic,
                mono_depths=mono_depths if options["include_monodepth"] else None,""",
"""                mask_dir=mask_cache_dir,
                mono_cache=depth_cache_dir,
                vo_recon=vo,
                imagery_root=imagery_root_path,
                include_plane_sweep=False,
                mono_depths=mono_depths if options["include_monodepth"] else None,""", content)

# 12. _run_inputs_fingerprint signature
content = replace_exact("""    dataset_dir: Optional[Path],
    imagery_root: Optional[Path],
    reference_dtm: Optional[str],
    strict_production: bool,
    allow_synthetic: bool,
) -> str:""",
"""    dataset_dir: Optional[Path],
    imagery_root: Optional[Path],
    reference_dtm: Optional[str],
) -> str:""", content)

content = replace_exact("""    payload = {
        "bbox": bbox,
        "dataset_dir": str(dataset_dir) if dataset_dir else None,
        "imagery_root": str(imagery_root) if imagery_root else None,
        "reference_dtm": reference_dtm,
        "synthetic_policy": {
            "allow_synthetic": allow_synthetic,
            "strict_production": strict_production,
        },
    }""",
"""    payload = {
        "bbox": bbox,
        "dataset_dir": str(dataset_dir) if dataset_dir else None,
        "imagery_root": str(imagery_root) if imagery_root else None,
        "reference_dtm": reference_dtm,
    }""", content)

# 13. typer run definition
content = re.sub(r'\s*allow_synthetic:\s*bool\s*=\s*typer\.Option\(\s*False,\s*help="Permit synthetic/heuristic fallbacks for development or smoke tests.",\s*\),', '', content, flags=re.DOTALL)
content = re.sub(r'\s*strict_production:\s*bool\s*=\s*typer\.Option\(\s*True,\s*help="Fail when production prerequisites are missing.",\s*\),', '', content, flags=re.DOTALL)
content = re.sub(r'\s*vo_force_synthetic:\s*bool\s*=\s*typer\.Option\(\s*False,\s*help="Force the legacy synthetic VO path \(skip imagery-backed VO\).",\s*\),', '', content, flags=re.DOTALL)

content = replace_exact("""        reference_dtm=reference_dtm,
        reference_nodata_values=reference_nodata_values,
        allow_synthetic=allow_synthetic,
        strict_production=strict_production,
        use_learned_uncertainty=use_learned_uncertainty,""",
"""        reference_dtm=reference_dtm,
        reference_nodata_values=reference_nodata_values,
        use_learned_uncertainty=use_learned_uncertainty,""", content)

content = replace_exact("""        legacy_vo=legacy_vo,
        vo_force_synthetic=vo_force_synthetic,
        vo_min_inliers=vo_min_inliers,""",
"""        legacy_vo=legacy_vo,
        vo_min_inliers=vo_min_inliers,""", content)

with open('cli/pipeline.py', 'w') as f:
    f.write(content)

print("Refactored pipeline safely!")
