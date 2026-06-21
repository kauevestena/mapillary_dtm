"""COLMAP reconstruction scaffolding."""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Mapping, Optional

import numpy as np

from .. import constants
from ..common_core import FrameMeta, Pose, ReconstructionResult
from .colmap_adapter import COLMAPConfig, COLMAPRunner, COLMAPUnavailable

logger = logging.getLogger(__name__)


def run(
    seqs: Mapping[str, List[FrameMeta]],
    rng_seed: int = 4025,
    refine_cameras: bool = False,
    refinement_method: str = "full",
    *,
    threads: Optional[int] = None,
    use_gpu: Optional[bool] = None,
    fixture_path: Optional[str | os.PathLike[str]] = None,
    force: bool = False,
    workspace_root: Optional[str | os.PathLike[str]] = None,
    imagery_root: Optional[str | os.PathLike[str]] = None,
    progress: bool = False,
) -> Dict[str, ReconstructionResult]:
    """
    Attempt to use a real COLMAP reconstruction (fixture or binary).
    """

    if not seqs:
        return {}

    if fixture_path is None:
        fixture_env = os.getenv("COLMAP_FIXTURE")
        if fixture_env:
            fixture_path = fixture_env

    threads = _select_threads(threads)
    use_gpu = _select_use_gpu(use_gpu)

    config = COLMAPConfig.from_kwargs(
        threads=threads,
        use_gpu=use_gpu,
        workspace_root=workspace_root,
    )
    runner = COLMAPRunner(config=config, workspace_root=workspace_root)
    try:
        base_results = runner.reconstruct(
            seqs,
            fixture_path=fixture_path,
            force=force,
            imagery_root=imagery_root,
            progress=progress,
        )
        if base_results:
            logger.info(
                "COLMAP adapter produced results using %s",
                fixture_path or "binary invocation",
            )
            return base_results
    except Exception as exc:  # pragma: no cover - unexpected adapter failure
        logger.exception("COLMAP adapter failed: %s", exc)
        raise

    raise RuntimeError("COLMAP failed or was not invoked")


def _select_threads(explicit: Optional[int]) -> int:
    if explicit is not None:
        return explicit
    env_value = os.getenv("COLMAP_THREADS")
    if env_value:
        try:
            return max(1, int(env_value))
        except ValueError:
            logger.warning(
                "Invalid COLMAP_THREADS value '%s'; using default %d",
                env_value,
                constants.COLMAP_DEFAULT_THREADS,
            )
    return constants.COLMAP_DEFAULT_THREADS


def _select_use_gpu(explicit: Optional[bool]) -> bool:
    if explicit is not None:
        return explicit
    env_value = os.getenv("COLMAP_USE_GPU")
    if env_value is not None:
        return env_value.strip() in {"1", "true", "TRUE", "True"}
    return constants.COLMAP_USE_GPU
