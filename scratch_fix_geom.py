import re

def replace_exact(filename, old, new):
    with open(filename, 'r') as f:
        c = f.read()
    if old not in c:
        print(f"Warning: could not find exact text in {filename}:\n{old}")
    else:
        c = c.replace(old, new)
        with open(filename, 'w') as f:
            f.write(c)
            
# 1. opensfm
replace_exact('geom/sfm_opensfm.py', """        except OpenSfMUnavailable as exc:
            if not allow_synthetic:
                raise
            logger.info("OpenSfM unavailable: %s; using synthetic fallback", exc)
        except Exception as exc:
            if not allow_synthetic:
                raise
            logger.exception("OpenSfM adapter failed: %s; falling back to synthetic path", exc)

    return _run_synthetic(
        seqs,
        rng_seed=rng_seed,
        refine_cameras=refine_cameras,
        refinement_method=refinement_method,
    )""", """        except Exception as exc:
            logger.error("OpenSfM unavailable: %s", exc)
            raise

    raise RuntimeError("OpenSfM failed or was not invoked")""")

# 2. colmap
replace_exact('geom/sfm_colmap.py', """        except COLMAPUnavailable as exc:
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

    return _run_synthetic(
        seqs,
        rng_seed=rng_seed,
        refine_cameras=refine_cameras,
        refinement_method=refinement_method,
    )""", """        except Exception as exc:
            logger.error("COLMAP unavailable: %s", exc)
            raise

    raise RuntimeError("COLMAP failed or was not invoked")""")

# 3. vo_simplified
replace_exact('geom/vo_simplified.py', """    if cv2 is None:
        raise RuntimeError("OpenCV is unavailable")
        if force_synthetic:
            log.info("VO forced to synthetic mode (flag or CLI)")
        else:
            log.info("OpenCV not available; using synthetic VO path")
        return _run_synthetic(seqs, rng_seed=rng_seed)""", """    if cv2 is None:
        raise RuntimeError("OpenCV is unavailable")""")

with open('geom/vo_simplified.py', 'r') as f:
    c = f.read()
c = re.sub(r'        except Exception as exc:\n\s*if not allow_synthetic:\n\s*raise\n\s*log.warning.*?used_synthetic = True', r'        except Exception as exc:\n            log.error("VO sequence failed: %s", exc)\n            raise', c, flags=re.DOTALL)
with open('geom/vo_simplified.py', 'w') as f:
    f.write(c)

print("Fixed geom modules.")
