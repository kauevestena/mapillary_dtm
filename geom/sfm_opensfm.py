"""
OpenSfM reconstruction stack (independent of COLMAP). No Mapillary seeding.
"""
from __future__ import annotations
from typing import Dict, List, Tuple

def run(seqs: dict) -> dict:
    """
    Execute an OpenSfM reconstruction per sequence or per AOI batch.
    Returns poses and sparse clouds (up-to-scale before height/scale solve).
    """
    # Placeholder
    return {}
