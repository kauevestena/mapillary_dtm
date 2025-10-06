"""
Filter sequences to car-only windows using raw geometry speed, and quality gates.
"""
from __future__ import annotations
from typing import Dict, List
from .. import constants
from ..common_core import FrameMeta

def filter_car_sequences(seqs: dict[str, list[FrameMeta]]) -> dict[str, list[FrameMeta]]:
    """
    Keep frames where windowed speed is within [MIN_SPEED_KMH, MAX_SPEED_KMH].
    Drop frames with quality_score < threshold.
    Keep all camera types allowed in constants.ALLOW_CAMERA_TYPES.
    """
    # Placeholder. Implement speed from successive geometry & timestamps.
    return seqs
