"""Deep-Image-Matching pipeline stage wrapper."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

from .. import constants
from ..common_core import FrameMeta, ReconstructionResult
from .dim_adapter import DIMRunner, DIMUnavailable

log = logging.getLogger(__name__)

def run(
    sequences: Mapping[str, Iterable[FrameMeta]],
    *,
    imagery_root: Optional[Path | str] = None,
    workspace_root: Optional[Path | str] = None,
    progress: bool = False,
) -> Dict[str, ReconstructionResult]:
    
    workspace = Path(workspace_root) if workspace_root else Path(constants.MAPILLARY_CACHE_ROOT) / "dim"
    runner = DIMRunner(
        workspace_root=workspace,
        extractor=constants.DIM_EXTRACTOR,
        matcher=constants.DIM_MATCHER,
    )
    
    if not runner.is_available():
        log.warning("DIM is not available. Skipping DIM reconstruction.")
        return {}
        
    return runner.reconstruct(sequences, imagery_root=imagery_root, progress=progress)
