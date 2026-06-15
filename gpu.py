"""
Centralized GPU detection and configuration utilities.

All GPU-aware modules should use this instead of ad-hoc device detection
to ensure consistent behavior across the pipeline.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger(__name__)

_DEVICE_CACHE: Optional[str] = None


def detect_device(*, prefer_gpu: Optional[bool] = None) -> str:
    """Return the best available compute device.

    Priority: explicit env ``DTM_DEVICE`` > CUDA > MPS > CPU.

    Parameters
    ----------
    prefer_gpu:
        When ``True``, CUDA / MPS are preferred even if ``GPU_AUTO_DETECT``
        is disabled.  When ``False``, always returns ``"cpu"``.  ``None``
        defers to the ``GPU_AUTO_DETECT`` constant and env var heuristics.
    """
    global _DEVICE_CACHE

    # Explicit env-var override always wins.
    env_device = os.getenv("DTM_DEVICE")
    if env_device:
        return env_device.strip()

    if prefer_gpu is False:
        return "cpu"

    if _DEVICE_CACHE is not None and prefer_gpu is None:
        return _DEVICE_CACHE

    device = _detect_device_impl(prefer_gpu)
    if prefer_gpu is None:
        _DEVICE_CACHE = device
    return device


def _detect_device_impl(prefer_gpu: Optional[bool]) -> str:
    try:
        import torch  # type: ignore
    except Exception:
        return "cpu"

    from . import constants

    auto_detect = getattr(constants, "GPU_AUTO_DETECT", True)
    if not auto_detect and prefer_gpu is not True:
        return "cpu"

    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
            log.info("GPU auto-detected: %s", name)
        except Exception:
            log.info("CUDA available (device name query failed)")
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        log.info("MPS backend available (Apple Silicon)")
        return "mps"

    return "cpu"


def get_optimal_dtype(device: str = "auto"):
    """Return the optimal inference dtype for *device*.

    FP16 is used on CUDA (significantly faster, negligible quality loss
    for inference).  FP32 on everything else.
    """
    try:
        import torch  # type: ignore
    except Exception:  # pragma: no cover
        return None  # torch unavailable; caller should handle

    if device == "auto":
        device = detect_device()

    if device.startswith("cuda"):
        return torch.float16
    return torch.float32


def auto_batch_size(
    *,
    vram_gb: Optional[float] = None,
    model_footprint_mb: float = 500,
    per_sample_mb: float = 50,
    min_batch: int = 1,
    max_batch: int = 16,
) -> int:
    """Estimate a reasonable batch size from available VRAM.

    Falls back to ``min_batch`` when CUDA is unavailable.
    """
    if vram_gb is None:
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                vram_gb = props.total_mem / (1024 ** 3)
            else:
                return min_batch
        except Exception:
            return min_batch

    available_mb = max(0, vram_gb * 1024 - model_footprint_mb - 512)  # keep 512 MB headroom
    batch = max(min_batch, int(available_mb / max(per_sample_mb, 1)))
    return min(batch, max_batch)


def configure_torch_defaults() -> None:
    """Apply global PyTorch performance knobs.

    Safe to call multiple times and when CUDA is unavailable.
    """
    try:
        import torch  # type: ignore
    except Exception:
        return

    # TF32 on Ampere+ gives ~2x matmul speedup with negligible precision loss.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
        torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
        log.debug("PyTorch: TF32 + cuDNN benchmark enabled")


def gpu_summary() -> dict[str, object]:
    """Return a JSON-friendly snapshot of the GPU environment."""
    info: dict[str, object] = {"device": detect_device()}
    try:
        import torch  # type: ignore

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["vram_gb"] = round(props.total_mem / (1024 ** 3), 1)
    except Exception:
        info["torch_version"] = None
        info["cuda_available"] = False
    return info
