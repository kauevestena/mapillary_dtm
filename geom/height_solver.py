"""
Per-sequence constant camera height (1–3 m) and metric scale solver.
Uses ground planes under the camera, GNSS deltas, and anchor footpoints.
"""
from __future__ import annotations
from typing import Dict, Tuple

def solve_scale_and_h(reconA: dict, reconB: dict, vo: dict, anchors: list, seqs: dict) -> tuple[dict, dict]:
    """
    Returns:
      - scales: dict[seq_id] -> float (metric scale)
      - heights: dict[seq_id] -> float (camera height h in meters)
    """
    # Placeholder: robust least squares with Huber/Tukey; enforce h ∈ [H_MIN_M, H_MAX_M].
    return {}, {}
