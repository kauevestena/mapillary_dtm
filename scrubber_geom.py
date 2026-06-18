import os
import re

def rewrite_file(filepath, replacements, regexes=[]):
    with open(filepath, 'r') as f:
        content = f.read()
    
    for old, new in replacements:
        if old not in content:
            print(f"FAILED to find literal in {filepath}:\n{old}")
            return False
        content = content.replace(old, new)
        
    for pattern, new in regexes:
        content = re.sub(pattern, new, content, flags=re.DOTALL)

    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Successfully rewritten {filepath}")
    return True

# anchors.py
rewrite_file("geom/anchors.py", [
    ("    allow_synthetic: bool = True,\n", ""),
    ("""            elif allow_synthetic:
                log.info("No real Mapillary anchors for %s; using synthetic placement", seq_id)
                observations = _synthetic_observations(frames, rng)
            else:
                log.info("No real/cached anchors for %s; continuing without synthetic anchors", seq_id)""", """            else:
                log.info("No real/cached anchors for %s; continuing without synthetic anchors", seq_id)"""),
    ("""        elif allow_synthetic:
            anchors = _synthesize_anchors(seq_id, frames)
            if anchors:
                _write_cache(cache_dir, seq_id, anchors)
        else:
            anchors = []
            log.info("No real/cached anchors for %s; continuing without synthetic anchors", seq_id)""", """        else:
            anchors = []
            log.info("No real/cached anchors for %s; continuing without synthetic anchors", seq_id)""")
])

# sfm_opensfm.py
rewrite_file("geom/sfm_opensfm.py", [
    ("    allow_synthetic: bool = True,\n", ""),
    ("""        except OpenSfMUnavailable as exc:
            if not allow_synthetic:
                raise
            logger.info("OpenSfM unavailable: %s; using synthetic fallback", exc)
        except Exception as exc:
            if not allow_synthetic:
                raise
            logger.exception("OpenSfM adapter failed: %s; falling back to synthetic path", exc)

    if not allow_synthetic:
        raise OpenSfMUnavailable("OpenSfM synthetic fallback disabled and no real reconstruction was produced")

    return _run_synthetic(
        seqs,
        rng_seed=rng_seed,
        refine_cameras=refine_cameras,
        refinement_method=refinement_method,
    )""", """        except Exception as exc:
            logger.error("OpenSfM unavailable: %s", exc)
            raise

    raise RuntimeError("OpenSfM failed or was not invoked")""")
], [
    (r'def _run_synthetic\(.*?(?=def run\()', '')
])

# sfm_colmap.py
rewrite_file("geom/sfm_colmap.py", [
    ("    allow_synthetic: bool = True,\n", ""),
    ("""        except COLMAPUnavailable as exc:
            if not allow_synthetic:
                raise
            logger.info("COLMAP unavailable: %s; using synthetic fallback", exc)
        except Exception as exc:
            if not allow_synthetic:
                raise
            logger.exception(
                "COLMAP adapter failed: %s; falling back to synthetic path",
                exc,
            )

    if not allow_synthetic:
        raise COLMAPUnavailable("COLMAP synthetic fallback disabled and no real reconstruction was produced")

    return _run_synthetic(
        seqs,
        rng_seed=rng_seed,
        refine_cameras=refine_cameras,
        refinement_method=refinement_method,
    )""", """        except Exception as exc:
            logger.error("COLMAP unavailable: %s", exc)
            raise

    raise RuntimeError("COLMAP failed or was not invoked")""")
], [
    (r'def _run_synthetic\(.*?(?=def run\()', '')
])

# vo_simplified.py
rewrite_file("geom/vo_simplified.py", [
    ("    force_synthetic: bool = False,\n", ""),
    ("    allow_synthetic: bool = True,\n", ""),
    ("""    if force_synthetic or cv2 is None:
        if not allow_synthetic:
            raise RuntimeError("VO synthetic fallback disabled and OpenCV/real VO is unavailable")
        if force_synthetic:
            log.info("VO forced to synthetic mode (flag or CLI)")
        else:
            log.info("OpenCV not available; using synthetic VO path")
        return _run_synthetic(seqs, rng_seed=rng_seed)""", """    if cv2 is None:
        raise RuntimeError("OpenCV is unavailable")"""),
    ("""        except Exception as exc:
            if not allow_synthetic:
                raise
            log.warning("VO sequence %s failed (%s); falling back to synthetic", seq.id, exc)
            fallback = _run_synthetic([seq], rng_seed=rng_seed)[seq.id]
            res[seq.id] = fallback
            used_synthetic = True""", """        except Exception as exc:
            log.error("VO sequence failed: %s", exc)
            raise"""),
    ("""    if used_synthetic:
        log.info("Some sequences used synthetic VO.")""", "")
], [
    (r'def _run_synthetic\(.*?(?=def run\()', '')
])

