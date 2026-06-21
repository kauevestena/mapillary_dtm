"""OpenSfM reconstruction scaffolding."""

from __future__ import annotations

import logging
import os
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np

from ..common_core import FrameMeta, Pose, ReconstructionResult
from .opensfm_adapter import OpenSfMRunner, OpenSfMUnavailable

logger = logging.getLogger(__name__)


def run(
    seqs: Mapping[str, List[FrameMeta]],
    rng_seed: int = 2025,
    refine_cameras: bool = False,
    refinement_method: str = "full",
    *,
    imagery_root: Optional[str | os.PathLike[str]] = None,
    workspace_root: Optional[str | os.PathLike[str]] = None,
    force: bool = False,
    progress: bool = False,
) -> Dict[str, ReconstructionResult]:
    """
    Attempt to use a real OpenSfM reconstruction (fixture or binary).
    """

    if not seqs:
        return {}

    runner = OpenSfMRunner(workspace_root=workspace_root)
    fixture = os.getenv("OPEN_SFM_FIXTURE")
    try:
        results = runner.reconstruct(
            seqs,
            fixture_path=fixture,
            force=force,
            imagery_root=imagery_root,
            progress=progress,
        )
        if results:
            logger.info("OpenSfM adapter produced results using %s", fixture or "binary invocation")
            return results
    except Exception as exc:
        logger.error("OpenSfM unavailable: %s", exc)
        raise

    raise RuntimeError("OpenSfM failed or was not invoked")
