"""
Monocular depth prediction scaffold.

This module synthetically produces low-resolution depth maps that mimic a
ground-aligned scene. Results are cached to ``cache/depth_mono`` so later
pipeline stages can densify sparse reconstructions without accessing a real
network model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from ..common_core import FrameMeta

CacheResult = Dict[str, Dict[str, Dict[str, np.ndarray]]]


def predict_depths(
    seqs: Mapping[str, Sequence[FrameMeta]],
    out_dir: Path | str = Path("cache/depth_mono"),
    resolution: tuple[int, int] = (96, 160),
    force: bool = False,
    seed: int = 1729,
) -> CacheResult:
    """Return (and cache) synthetic per-frame depth/uncertainty maps.

    Parameters
    ----------
    seqs:
        Mapping of sequence id to ordered :class:`FrameMeta` entries.
    out_dir:
        Destination directory for cached ``.npz`` files.
    resolution:
        Target (rows, cols) of the generated depth maps. The aspect ratio is
        preserved relative to the frame intrinsics when possible.
    force:
        When ``True`` existing cache files are ignored and regenerated.
    seed:
        Base RNG seed used to keep outputs deterministic across runs.
    """

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results: CacheResult = {}
    rng = np.random.default_rng(seed)

    for seq_id, frames in seqs.items():
        if not frames:
            continue

        frame_results: Dict[str, Dict[str, np.ndarray]] = {}
        for index, frame in enumerate(frames):
            cache_path = out_path / f"{frame.image_id}.npz"
            depth: np.ndarray | None = None
            uncert: np.ndarray | None = None

            if cache_path.exists() and not force:
                depth, uncert = _load_cached_depth(cache_path)

            if depth is None or uncert is None:
                depth, uncert = _synthesize_depth(
                    frame,
                    resolution=resolution,
                    rng=rng,
                    frame_index=index,
                )
                _write_depth(cache_path, depth, uncert)

            frame_results[frame.image_id] = {
                "depth": depth.astype(np.float32, copy=False),
                "uncertainty": uncert.astype(np.float32, copy=False),
            }

        if frame_results:
            results[seq_id] = frame_results

    return results


def _load_cached_depth(path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        with np.load(path) as data:
            depth = np.asarray(data.get("depth"), dtype=np.float32)
            uncert = np.asarray(data.get("uncertainty"), dtype=np.float32)
            if depth.ndim != 2 or depth.size == 0:
                return None, None
            if uncert.shape != depth.shape:
                uncert = np.full_like(depth, 0.25, dtype=np.float32)
            return depth, uncert
    except Exception:
        return None, None


def _synthesize_depth(
    frame: FrameMeta,
    resolution: tuple[int, int],
    rng: np.random.Generator,
    frame_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    params = frame.cam_params or {}
    width = float(params.get("width") or params.get("image_width") or 2048.0)
    height = float(params.get("height") or params.get("image_height") or 1536.0)
    aspect = width / max(height, 1.0)

    rows_target, cols_target = resolution
    cols = int(round(min(cols_target, rows_target * aspect)))
    rows = int(round(max(rows_target // 2, rows_target)))
    rows = max(16, rows)
    cols = max(16, cols)

    v = np.linspace(0.05, 1.0, rows, dtype=np.float32)
    tilt = float(rng.normal(scale=0.05))
    ground_depth = 8.0 + 20.0 * (1.0 - v)  # farther near horizon
    ground_depth *= (1.0 + tilt * (v - 0.5))
    depth = np.repeat(ground_depth[:, None], cols, axis=1)

    # Introduce gentle undulation to mimic small bumps/curbs.
    noise_rng = np.random.default_rng((abs(hash(frame.image_id)) + frame_index) & 0xFFFF)
    perturb = noise_rng.normal(scale=0.25, size=depth.shape).astype(np.float32)
    depth = depth + perturb
    depth = np.clip(depth, 3.0, 60.0)

    # Confidence decreases near horizon and for noisy pixels.
    norm_row = v[:, None]
    uncert = np.repeat(norm_row, cols, axis=1)
    uncert = 0.1 + 0.4 * uncert
    uncert += np.abs(perturb) * 0.02
    uncert = np.clip(uncert, 0.1, 0.6).astype(np.float32)

    return depth.astype(np.float32), uncert.astype(np.float32)


def _write_depth(path: Path, depth: np.ndarray, uncert: np.ndarray) -> None:
    try:
        np.savez_compressed(path, depth=depth, uncertainty=uncert)
    except OSError:
        # Failing to cache should not break the pipeline; the caller can retry.
        pass
