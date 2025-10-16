"""
Monocular depth prediction scaffold with optional real-model integration.

The default implementation synthesizes plausible ground-aligned depth maps.
When a supported adapter is available (e.g., PyTorch-based MiDaS weights),
the adapter is invoked to produce per-frame depth and uncertainty estimates.
Results are cached to ``cache/depth_mono`` so later pipeline stages can
reuse them without recomputation.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..common_core import FrameMeta
from ..ingest.image_loader import ImageryLoader

try:  # Optional dependency for real depth inference
    import torch  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    torch = None

log = logging.getLogger(__name__)

CacheResult = Dict[str, Dict[str, Dict[str, np.ndarray]]]


class _DepthAdapter:
    """Interface for pluggable depth backends."""

    def predict(self, frame: FrameMeta) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError


class _TorchDepthAdapter(_DepthAdapter):
    def __init__(self, model_path: Path, device: str, imagery_root: Optional[Path | str]) -> None:
        if torch is None:  # pragma: no cover - handled upstream
            raise RuntimeError("Torch not available for depth adapter")

        self.device = torch.device(device)
        self.loader = ImageryLoader(imagery_root)
        self.model = torch.jit.load(str(model_path), map_location=self.device)
        self.model.eval()

    def predict(self, frame: FrameMeta) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        image = self.loader.load_rgb(frame)
        if image is None:
            return None

        tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        else:
            tensor = tensor.permute(2, 0, 1)
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            prediction = self.model(tensor)

        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]

        depth = prediction.squeeze().detach().cpu().numpy().astype(np.float32, copy=False)
        if depth.ndim != 2:
            depth = depth.reshape(depth.shape[-2], depth.shape[-1])
        depth = _normalize_depth(depth)
        uncertainty = _estimate_uncertainty(depth)
        return depth, uncertainty


def predict_depths(
    seqs: Mapping[str, Sequence[FrameMeta]],
    out_dir: Path | str = Path("cache/depth_mono"),
    resolution: tuple[int, int] = (96, 160),
    force: bool = False,
    seed: int = 1729,
    adapter: Optional[_DepthAdapter] = None,
    imagery_root: Optional[Path | str] = None,
    use_gpu: Optional[bool] = None,
) -> CacheResult:
    """Return (and cache) depth/uncertainty maps for each frame."""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if adapter is None:
        adapter = _init_default_adapter(imagery_root=imagery_root, use_gpu=use_gpu)

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

            if (depth is None or uncert is None) and adapter is not None:
                try:
                    prediction = adapter.predict(frame)
                except Exception as exc:  # pragma: no cover - adapter failure
                    log.warning("Depth adapter failed for %s: %s", frame.image_id, exc)
                    prediction = None
                if prediction is not None:
                    depth, uncert = prediction
                    depth = depth.astype(np.float32, copy=False)
                    uncert = uncert.astype(np.float32, copy=False)

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


def _init_default_adapter(
    imagery_root: Optional[Path | str],
    use_gpu: Optional[bool],
) -> Optional[_DepthAdapter]:
    model_path = os.getenv("MONODEPTH_MODEL_PATH")
    if not model_path:
        return None

    model_file = Path(model_path)
    if not model_file.exists():
        log.warning("MONODEPTH_MODEL_PATH=%s does not exist; skipping adapter", model_path)
        return None

    if torch is None:
        log.warning("PyTorch not available; cannot load monodepth model")
        return None

    device_env = os.getenv("MONODEPTH_DEVICE")
    gpu_env = os.getenv("MONODEPTH_USE_GPU")
    gpu_requested = (
        use_gpu
        if use_gpu is not None
        else gpu_env is not None and gpu_env.lower() in {"1", "true", "yes"}
    )

    if device_env:
        device = device_env
    elif gpu_requested and torch.cuda.is_available():  # type: ignore[attr-defined]
        device = "cuda"
    else:
        if gpu_requested and not torch.cuda.is_available():  # type: ignore[attr-defined]
            log.info("MONODEPTH GPU requested but CUDA not available; using CPU")
        device = "cpu"

    try:
        return _TorchDepthAdapter(model_file, device, imagery_root)
    except Exception as exc:
        log.warning("Failed to initialize torch depth adapter: %s", exc)
        return None


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        depth = depth.reshape(depth.shape[-2], depth.shape[-1])
    depth = depth - np.nanmin(depth)
    max_val = float(np.nanmax(depth))
    if not np.isfinite(max_val) or max_val < 1e-6:
        max_val = 1.0
    depth = depth / max_val
    depth = 3.0 + depth * 40.0
    return depth.astype(np.float32)


def _estimate_uncertainty(depth: np.ndarray) -> np.ndarray:
    grad_y, grad_x = np.gradient(depth.astype(np.float32))
    mag = np.sqrt(grad_x**2 + grad_y**2)
    mag_norm = mag / (np.max(mag) + 1e-6)
    uncert = 0.1 + 0.4 * np.clip(mag_norm, 0.0, 1.0)
    return np.clip(uncert, 0.1, 0.6).astype(np.float32)


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
