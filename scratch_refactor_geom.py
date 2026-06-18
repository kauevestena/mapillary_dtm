import re

# 1. sfm_opensfm.py
with open('geom/sfm_opensfm.py', 'r') as f:
    content = f.read()

# remove def _run_synthetic and all its body
content = re.sub(r'def _run_synthetic\(.*?(?=def |class |$)', '', content, flags=re.DOTALL)
# remove allow_synthetic argument
content = re.sub(r'\s*allow_synthetic:\s*bool\s*=\s*True,', '', content)
# remove fallback block
opensfm_fallback = """        except Exception as exc:
            if not allow_synthetic:
                raise
            logger.info("OpenSfM unavailable: %s; using synthetic fallback", exc)
            return _run_synthetic(seqs, rng_seed=rng_seed)"""
opensfm_fallback_2 = """        except Exception as exc:
            if not allow_synthetic:
                raise
            logger.exception("OpenSfM adapter failed: %s; falling back to synthetic path", exc)
            return _run_synthetic(seqs, rng_seed=rng_seed)"""
content = content.replace(opensfm_fallback, """        except Exception as exc:
            logger.error("OpenSfM unavailable: %s", exc)
            raise""")
content = content.replace(opensfm_fallback_2, """        except Exception as exc:
            logger.exception("OpenSfM adapter failed: %s", exc)
            raise""")
content = re.sub(r'\s*if not allow_synthetic:\n\s*raise OpenSfMUnavailable\("OpenSfM synthetic fallback disabled and no real reconstruction was produced"\)', '', content)
content = re.sub(r'\s*return _run_synthetic\(seqs, rng_seed=rng_seed\)', '', content)
# remove synthetic imports
content = re.sub(r',\s*synthetic_ground_offsets', '', content)

with open('geom/sfm_opensfm.py', 'w') as f:
    f.write(content)

# 2. sfm_colmap.py
with open('geom/sfm_colmap.py', 'r') as f:
    content = f.read()

content = re.sub(r'def _run_synthetic\(.*?(?=def |class |$)', '', content, flags=re.DOTALL)
content = re.sub(r'\s*allow_synthetic:\s*bool\s*=\s*True,', '', content)

colmap_fallback = """        except Exception as exc:
            if not allow_synthetic:
                raise
            logger.info("COLMAP unavailable: %s; using synthetic fallback", exc)
            return _run_synthetic(seqs, rng_seed=rng_seed)"""
colmap_fallback_2 = """        except Exception as exc:
            if not allow_synthetic:
                raise
            logger.exception(
                "COLMAP adapter failed: %s; falling back to synthetic path",
                exc,
            )
            return _run_synthetic(seqs, rng_seed=rng_seed)"""
content = content.replace(colmap_fallback, """        except Exception as exc:
            logger.error("COLMAP unavailable: %s", exc)
            raise""")
content = content.replace(colmap_fallback_2, """        except Exception as exc:
            logger.exception("COLMAP adapter failed: %s", exc)
            raise""")
content = re.sub(r'\s*if not allow_synthetic:\n\s*raise COLMAPUnavailable\("COLMAP synthetic fallback disabled and no real reconstruction was produced"\)', '', content)
content = re.sub(r'\s*return _run_synthetic\(seqs, rng_seed=rng_seed\)', '', content)
content = re.sub(r',\s*synthetic_ground_offsets', '', content)

with open('geom/sfm_colmap.py', 'w') as f:
    f.write(content)

# 3. vo_simplified.py
with open('geom/vo_simplified.py', 'r') as f:
    content = f.read()

content = re.sub(r'def _run_synthetic\(.*?(?=def |class |$)', '', content, flags=re.DOTALL)
content = re.sub(r'\s*force_synthetic:\s*bool\s*=\s*False,', '', content)
content = re.sub(r'\s*allow_synthetic:\s*bool\s*=\s*True,', '', content)
content = re.sub(r'\s*allow_synthetic_steps:\s*bool\s*=\s*True,', '', content)
content = re.sub(r'\s*allow_synthetic_steps=allow_synthetic,', '', content)

vo_fallback = """    if force_synthetic or cv2 is None:
        if not allow_synthetic:
            raise RuntimeError("VO synthetic fallback disabled and OpenCV/real VO is unavailable")
        if force_synthetic:
            log.info("VO forced to synthetic mode (flag or CLI)")
        else:
            log.info("OpenCV not available; using synthetic VO path")
        return _run_synthetic(seqs, rng_seed=rng_seed)"""
content = content.replace(vo_fallback, """    if cv2 is None:
        raise RuntimeError("OpenCV is unavailable")""")

content = re.sub(r'        except Exception as exc:\n            if not allow_synthetic:\n                raise\n            log\.warning\("VO sequence .*? failed \(\%s\); falling back to synthetic", exc\)\n            fallback = _run_synthetic\(\[seq\], rng_seed=rng_seed\)\[seq\.id\]\n            res\[seq\.id\] = fallback\n            used_synthetic = True', r'        except Exception as exc:\n            log.error("VO sequence failed: %s", exc)\n            raise', content)

content = re.sub(r'\s*if used_synthetic:\n\s*log.info\("Some sequences used synthetic VO."\)', '', content)
content = re.sub(r',\s*synthetic_ground_offsets', '', content)

with open('geom/vo_simplified.py', 'w') as f:
    f.write(content)

# 4. depth/monodepth.py
with open('depth/monodepth.py', 'r') as f:
    content = f.read()

content = re.sub(r'\s*allow_synthetic:\s*bool\s*=\s*True,', '', content)
content = re.sub(r'\s*allow_synthetic=allow_synthetic,', '', content)
content = re.sub(r'def _synthetic_depth_map\(.*?(?=def |class |$)', '', content, flags=re.DOTALL)

mono_fallback = """                if _should_init_model_mono(model_path, allow_synthetic=allow_synthetic):
                    from ..ml.monodepth_adapter import MiDaSAdapter
                    model = MiDaSAdapter(model_path=model_path, device=os.environ.get("MONODEPTH_DEVICE", "cpu"))
                else:
                    model = None"""
mono_new = """                from ..ml.monodepth_adapter import MiDaSAdapter
                model = MiDaSAdapter(model_path=model_path, device=os.environ.get("MONODEPTH_DEVICE", "cpu"))"""
content = content.replace(mono_fallback, mono_new)

mono_syn_block = """                if model is None:
                    if not allow_synthetic:
                        raise RuntimeError(
                            "Monodepth missing and synthetic disabled. "
                            "Set MONODEPTH_MODEL_PATH to a valid torchscript model."
                        )
                    dmap = _synthetic_depth_map(w, h, rng)
                    # use slightly less reliable quality for synthetic
                    metadata["quality"] = 0.6
                else:"""
mono_syn_new = """                if model is None:
                    raise RuntimeError("Monodepth model failed to initialize.")
                else:"""
content = content.replace(mono_syn_block, mono_syn_new)

# Remove `_should_init_model_mono` usages since we'll always load the real model
content = re.sub(r'def _should_init_model_mono\(model_path, allow_synthetic: bool = True\):.*?(?=def |class |$)', '', content, flags=re.DOTALL)

with open('depth/monodepth.py', 'w') as f:
    f.write(content)

# 5. semantics/ground_masks.py
with open('semantics/ground_masks.py', 'r') as f:
    content = f.read()

content = re.sub(r'\s*allow_heuristic:\s*bool\s*=\s*True,', '', content)
content = re.sub(r'\s*require_provenance\s*=\s*not allow_heuristic', 'require_provenance = True', content)
content = re.sub(r'def _heuristic_mask\(.*?(?=def |class |$)', '', content, flags=re.DOTALL)
content = re.sub(r'def _should_init_model_masker\(.*?(?=def |class |$)', '', content, flags=re.DOTALL)

mask_fallback = """                if model is None:
                    if _should_init_model_masker(model_path, allow_heuristic=allow_heuristic):
                        from ..ml.segmentation_adapter import SegmentationAdapter
                        model = SegmentationAdapter(model_path=model_path, device=os.environ.get("SEGMENTATION_DEVICE", "cpu"))
                    else:
                        model = None"""
mask_new = """                if model is None:
                    from ..ml.segmentation_adapter import SegmentationAdapter
                    model = SegmentationAdapter(model_path=model_path, device=os.environ.get("SEGMENTATION_DEVICE", "cpu"))"""
content = content.replace(mask_fallback, mask_new)

mask_syn_block = """                if model is None:
                    if not allow_heuristic:
                        raise RuntimeError(
                            "Ground mask missing and heuristic masks are disabled. "
                            "Set SEGMENTATION_MODEL_PATH or run with --allow-heuristic"
                        )
                    prob_map = _heuristic_mask(w, h, rng)
                    metadata = {
                        "source_type": "heuristic",
                        "quality": 0.5,
                        "generated_at": datetime.now(timezone.utc).isoformat()
                    }
                else:"""
mask_syn_new = """                if model is None:
                    raise RuntimeError("Ground mask model missing.")
                else:"""
content = content.replace(mask_syn_block, mask_syn_new)

with open('semantics/ground_masks.py', 'w') as f:
    f.write(content)

# 6. geom/anchors.py
with open('geom/anchors.py', 'r') as f:
    content = f.read()

content = re.sub(r'\s*allow_synthetic:\s*bool\s*=\s*True,', '', content)
content = re.sub(r'def _synthetic_observations\(.*?(?=def |class |$)', '', content, flags=re.DOTALL)

anchor_fallback = """        if not points_3d:
            if not map_client:
                log.info("Mapillary API client not provided; skipping vector tiles search")
            elif allow_synthetic:
                log.info("No real Mapillary anchors for %s; using synthetic placement", seq_id)
                observations = _synthetic_observations(frames, rng)
            else:
                log.info("No real/cached anchors for %s; continuing without synthetic anchors", seq_id)"""
anchor_new = """        if not points_3d:
            if not map_client:
                log.info("Mapillary API client not provided; skipping vector tiles search")
            else:
                log.info("No real/cached anchors for %s; continuing without synthetic anchors", seq_id)"""
content = content.replace(anchor_fallback, anchor_new)

content = re.sub(r'\s*elif source == "synthetic":\n\s*observations = _synthetic_observations\(frames, rng\)', '', content)
content = re.sub(r'"""Synthetic anchor discovery using cached detections or heuristics."""', '"""Anchor discovery using cached detections."""', content)

with open('geom/anchors.py', 'w') as f:
    f.write(content)

print("Refactored modules.")
