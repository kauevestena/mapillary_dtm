"""Utilities to produce (or load) per-image ground probability masks."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

import numpy as np

from ..common_core import FrameMeta

log = logging.getLogger(__name__)


def prepare(
    seqs: Mapping[str, List[FrameMeta]],
    out_dir: Path | str = Path("cache/masks"),
    backend: str = "soft-horizon",
    force: bool = False,
) -> Dict[str, List[Path]]:
    """Generate ground-probability masks for each frame.

    Parameters
    ----------
    seqs:
        Mapping of sequence id to ordered list of :class:`FrameMeta`.
    out_dir:
        Destination directory where `.npz` mask files will be written.
    backend:
        Strategy used to synthesize a probability map when no pre-computed
        mask exists. Currently supports ``"soft-horizon"`` (default) and
        ``"constant"``.
    force:
        When ``True`` the mask will be regenerated even if an on-disk file
        already exists.

    Returns
    -------
    dict
        Mapping of sequence id to list of mask file paths (aligned with the
        input frame order).
    """

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, List[Path]] = {}
    for seq_id, frames in seqs.items():
        mask_paths: List[Path] = []
        for frame in frames:
            mask_path = out_path / f"{frame.image_id}.npz"
            if mask_path.exists() and not force:
                mask_paths.append(mask_path)
                continue

            prob = None
            if not force:
                prob = _load_existing_mask(mask_path)
            if prob is None:
                prob = _synthesize_mask(frame, backend=backend)

            _write_mask(mask_path, prob, frame)
            mask_paths.append(mask_path)
        summary[seq_id] = mask_paths

    return summary


def _load_existing_mask(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        with np.load(path) as data:
            if "prob" in data:
                arr = np.asarray(data["prob"], dtype=np.float32)
                if arr.ndim == 2:
                    return arr
    except Exception as exc:  # pragma: no cover - corrupted files rare
        log.warning("Failed to load mask %s: %s", path, exc)
    return None


def _synthesize_mask(frame: FrameMeta, backend: str = "soft-horizon") -> np.ndarray:
    shape = _mask_shape(frame)
    if backend == "constant":
        return np.full(shape, 0.75, dtype=np.float32)
    if backend == "soft-horizon":
        # Simple heuristic: higher ground probability near the bottom of the image.
        rows = np.linspace(0.0, 1.0, shape[0], dtype=np.float32)
        prob = np.repeat(rows[:, None], shape[1], axis=1)
        return np.clip(prob ** 0.5, 0.0, 1.0)
    raise ValueError(f"Unknown mask backend '{backend}'")


def _mask_shape(frame: FrameMeta, target_width: int = 256) -> tuple[int, int]:
    params = frame.cam_params or {}
    width = _safe_int(params.get("width") or params.get("image_width"))
    height = _safe_int(params.get("height") or params.get("image_height"))
    if width and height and width > 0 and height > 0:
        scale = min(1.0, target_width / float(width))
        w = max(1, int(round(width * scale)))
        h = max(1, int(round(height * scale)))
        return h, w
    return (128, 128)


def _write_mask(path: Path, prob: np.ndarray, frame: FrameMeta) -> None:
    try:
        np.savez_compressed(
            path,
            prob=np.clip(prob.astype(np.float32), 0.0, 1.0),
            image_id=frame.image_id,
            seq_id=frame.seq_id,
            captured_at_ms=frame.captured_at_ms,
        )
    except OSError as exc:  # pragma: no cover - filesystem error
        log.error("Failed to save mask %s: %s", path, exc)
        raise


def _safe_int(value) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
