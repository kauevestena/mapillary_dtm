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
from contextlib import nullcontext
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
DEFAULT_MONODEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"


class _DepthAdapter:
    """Interface for pluggable depth backends."""

    def predict(self, frame: FrameMeta) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError

    def provenance(self) -> dict[str, str | None]:
        return {
            "source_type": "model",
            "backend": self.__class__.__name__,
            "model_id": None,
            "model_revision": None,
        }


class _TorchDepthAdapter(_DepthAdapter):
    def __init__(self, model_path: Path, device: str, imagery_root: Optional[Path | str]) -> None:
        if torch is None:  # pragma: no cover - handled upstream
            raise RuntimeError("Torch not available for depth adapter")

        self.device = torch.device(device)
        self.loader = ImageryLoader(imagery_root)
        self.model = torch.jit.load(str(model_path), map_location=self.device)
        self.model.eval()
        self.path = model_path

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

    def provenance(self) -> dict[str, str | None]:
        return {
            "source_type": "model",
            "backend": "torchscript",
            "model_id": str(self.path),
            "model_revision": None,
        }


class _TransformersDepthAdapter(_DepthAdapter):
    def __init__(self, model_id: str, device: str, imagery_root: Optional[Path | str]) -> None:
        if torch is None:  # pragma: no cover - handled upstream
            raise RuntimeError("Torch not available for transformers depth adapter")
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("transformers is not available for depth adapter") from exc

        self.device = torch.device(device)
        self.loader = ImageryLoader(imagery_root)
        self.model_id = model_id
        self.revision = os.getenv("MONODEPTH_MODEL_REVISION")
        self.cache_dir = os.getenv("DTM_MODEL_CACHE_DIR", "models/huggingface")
        local_only = os.getenv("DTM_MODELS_LOCAL_ONLY", "1").lower() not in {"0", "false", "no"}
        self.processor = AutoImageProcessor.from_pretrained(
            model_id,
            revision=self.revision,
            cache_dir=self.cache_dir,
            local_files_only=local_only,
        )
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_id,
            revision=self.revision,
            cache_dir=self.cache_dir,
            local_files_only=local_only,
        ).to(self.device)
        self.model.eval()

    def predict(self, frame: FrameMeta) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        image = self.loader.load_rgb(frame)
        if image is None:
            return None
        try:
            from PIL import Image
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Pillow is required for depth inference") from exc

        pil_image = Image.fromarray(image)
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
            depth = outputs.predicted_depth
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=(pil_image.height, pil_image.width),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0).squeeze(0)
        arr = depth.detach().cpu().numpy().astype(np.float32, copy=False)
        arr = np.where(np.isfinite(arr), arr, np.nan).astype(np.float32)
        if not np.isfinite(arr).any():
            return None
        arr = np.clip(arr, 0.5, 120.0).astype(np.float32)
        uncertainty = _estimate_uncertainty(arr)
        return arr, uncertainty

    def provenance(self) -> dict[str, str | None]:
        return {
            "source_type": "model",
            "backend": "transformers-depth-anything",
            "model_id": self.model_id,
            "model_revision": self.revision,
        }


def predict_depths(
    seqs: Mapping[str, Sequence[FrameMeta]],
    out_dir: Path | str = Path("cache/depth_mono"),
    resolution: tuple[int, int] = (96, 160),
    force: bool = False,
    seed: int = 1729,
    adapter: Optional[_DepthAdapter] = None,
    imagery_root: Optional[Path | str] = None,
    use_gpu: Optional[bool] = None,
    allow_synthetic: bool = True,
    progress: bool | None = None,
) -> CacheResult:
    """Return (and cache) depth/uncertainty maps for each frame."""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    adapter_initialized = adapter is not None

    results: CacheResult = {}
    rng = np.random.default_rng(seed)
    require_provenance = not allow_synthetic
    cached_count = 0
    generated_count = 0
    total_frames = sum(len(frames) for frames in seqs.values())

    with _progress_bar(total_frames, "Monodepth", progress) as pbar:
        for seq_id, frames in seqs.items():
            if not frames:
                continue

            frame_results: Dict[str, Dict[str, np.ndarray]] = {}
            for index, frame in enumerate(frames):
                cache_path = out_path / f"{frame.image_id}.npz"
                depth: np.ndarray | None = None
                uncert: np.ndarray | None = None
                provenance: dict[str, str | None] | None = None

                if cache_path.exists() and not force:
                    cached_depth, cached_uncert, cached_provenance = _load_cached_depth(
                        cache_path,
                        require_provenance=require_provenance,
                    )
                    depth, uncert, provenance = cached_depth, cached_uncert, cached_provenance
                    if depth is not None and uncert is not None:
                        cached_count += 1
                        frame_results[frame.image_id] = {
                            "depth": depth.astype(np.float32, copy=False),
                            "uncertainty": uncert.astype(np.float32, copy=False),
                        }
                        _progress_update(
                            pbar,
                            cached=cached_count,
                            generated=generated_count,
                        )
                        continue

                if not adapter_initialized:
                    if _should_init_default_adapter(allow_synthetic=allow_synthetic):
                        adapter = _init_default_adapter(imagery_root=imagery_root, use_gpu=use_gpu)
                    adapter_initialized = True

                if adapter is not None:
                    try:
                        prediction = adapter.predict(frame)
                    except Exception as exc:  # pragma: no cover - adapter failure
                        log.warning("Depth adapter failed for %s: %s", frame.image_id, exc)
                        prediction = None
                    if prediction is not None:
                        depth, uncert = prediction
                        depth = depth.astype(np.float32, copy=False)
                        uncert = uncert.astype(np.float32, copy=False)
                        depth, uncert = _downsample_prediction(depth, uncert, resolution)
                        provenance = adapter.provenance()

                if depth is None or uncert is None:
                    if not allow_synthetic:
                        raise RuntimeError(
                            "Monodepth prediction unavailable and synthetic depth is disabled. "
                            "Provide provenanced cached depth maps, set MONODEPTH_MODEL_PATH, "
                            "or cache the configured Hugging Face depth model."
                        )
                    depth, uncert = _synthesize_depth(
                        frame,
                        resolution=resolution,
                        rng=rng,
                        frame_index=index,
                    )
                    provenance = {
                        "source_type": "synthetic",
                        "backend": "procedural",
                        "model_id": None,
                        "model_revision": None,
                    }

                _write_depth(cache_path, depth, uncert, provenance=provenance)
                generated_count += 1

                frame_results[frame.image_id] = {
                    "depth": depth.astype(np.float32, copy=False),
                    "uncertainty": uncert.astype(np.float32, copy=False),
                }
                _progress_update(
                    pbar,
                    cached=cached_count,
                    generated=generated_count,
                )

            if frame_results:
                results[seq_id] = frame_results

    return results


class _NoProgress:
    def __enter__(self) -> "_NoProgress":
        return self

    def __exit__(self, *args) -> None:
        return None

    def update(self, value: int = 1) -> None:
        return None

    def set_postfix(self, *args, **kwargs) -> None:
        return None


def _progress_bar(total: int, desc: str, enabled: bool | None):
    if enabled is None:
        try:
            enabled = os.isatty(2)
        except Exception:
            enabled = False
    if not enabled:
        return nullcontext(_NoProgress())
    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover - optional display dependency
        return nullcontext(_NoProgress())
    return tqdm(total=total, desc=desc, unit="img")


def _progress_update(pbar, *, cached: int, generated: int) -> None:
    try:
        pbar.update(1)
        current = int(getattr(pbar, "n", getattr(pbar, "updates", 0)) or 0)
        total = int(getattr(pbar, "total", 0) or 0)
        if current == 1 or current % 50 == 0 or (total and current >= total):
            pbar.set_postfix(cached=cached, generated=generated)
    except Exception:
        return None


def _load_cached_depth(
    path: Path,
    *,
    require_provenance: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, str | None] | None]:
    try:
        with np.load(path) as data:
            depth = np.asarray(data.get("depth"), dtype=np.float32)
            uncert = np.asarray(data.get("uncertainty"), dtype=np.float32)
            if depth.ndim != 2 or depth.size == 0:
                return None, None, None
            if uncert.shape != depth.shape:
                uncert = np.full_like(depth, 0.25, dtype=np.float32)
            provenance = {
                "source_type": _npz_scalar(data, "source_type"),
                "backend": _npz_scalar(data, "backend"),
                "model_id": _npz_scalar(data, "model_id"),
                "model_revision": _npz_scalar(data, "model_revision"),
            }
            if require_provenance and provenance["source_type"] not in {"model", "external"}:
                return None, None, None
            return depth, uncert, provenance
    except Exception:
        return None, None, None


def _init_default_adapter(
    imagery_root: Optional[Path | str],
    use_gpu: Optional[bool],
) -> Optional[_DepthAdapter]:
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

    model_path = os.getenv("MONODEPTH_MODEL_PATH")
    if model_path:
        model_file = Path(model_path)
        if not model_file.exists():
            log.warning("MONODEPTH_MODEL_PATH=%s does not exist; skipping adapter", model_path)
            return None
        try:
            return _TorchDepthAdapter(model_file, device, imagery_root)
        except Exception as exc:
            log.warning("Failed to initialize torch depth adapter: %s", exc)
            return None

    model_id = os.getenv("MONODEPTH_MODEL_ID", DEFAULT_MONODEPTH_MODEL_ID)
    if not model_id:
        return None
    try:
        return _TransformersDepthAdapter(model_id, device, imagery_root)
    except Exception as exc:
        log.warning("Failed to initialize transformers depth adapter: %s", exc)
        return None


def _should_init_default_adapter(*, allow_synthetic: bool) -> bool:
    if os.getenv("MONODEPTH_MODEL_PATH"):
        return True
    if not allow_synthetic:
        return True
    return "MONODEPTH_MODEL_ID" in os.environ and bool(os.getenv("MONODEPTH_MODEL_ID"))


def _downsample_prediction(
    depth: np.ndarray,
    uncert: np.ndarray,
    resolution: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample large model outputs to the cache resolution without upscaling."""
    depth = np.asarray(depth, dtype=np.float32)
    uncert = np.asarray(uncert, dtype=np.float32)
    if depth.ndim != 2:
        depth = depth.reshape(depth.shape[-2], depth.shape[-1])
    if uncert.shape != depth.shape:
        uncert = np.full_like(depth, 0.25, dtype=np.float32)
    target_rows, target_cols = resolution
    if depth.shape[0] <= target_rows and depth.shape[1] <= target_cols:
        return depth, uncert
    out_shape = (max(1, int(target_rows)), max(1, int(target_cols)))
    return (
        _resize_array(depth, out_shape).astype(np.float32, copy=False),
        _resize_array(uncert, out_shape).astype(np.float32, copy=False),
    )


def _resize_array(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    try:
        import cv2  # type: ignore

        return cv2.resize(
            arr.astype(np.float32, copy=False),
            (shape[1], shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    except Exception:
        try:
            from PIL import Image

            image = Image.fromarray(arr.astype(np.float32, copy=False), mode="F")
            image = image.resize((shape[1], shape[0]), resample=Image.Resampling.BILINEAR)
            return np.asarray(image, dtype=np.float32)
        except Exception:
            rows = np.linspace(0, arr.shape[0] - 1, shape[0]).astype(int)
            cols = np.linspace(0, arr.shape[1] - 1, shape[1]).astype(int)
            return arr[np.ix_(rows, cols)].astype(np.float32, copy=False)


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


def _write_depth(
    path: Path,
    depth: np.ndarray,
    uncert: np.ndarray,
    *,
    provenance: Mapping[str, str | None] | None = None,
) -> None:
    provenance = dict(provenance or {})
    try:
        np.savez_compressed(
            path,
            depth=depth,
            uncertainty=uncert,
            source_type=provenance.get("source_type") or "",
            backend=provenance.get("backend") or "",
            model_id=provenance.get("model_id") or "",
            model_revision=provenance.get("model_revision") or "",
        )
    except OSError:
        # Failing to cache should not break the pipeline; the caller can retry.
        pass


def _npz_scalar(data, key: str) -> str | None:
    if key not in data:
        return None
    value = data[key]
    if getattr(value, "shape", ()) == ():
        raw = value.item()
    else:
        raw = value
    if raw is None:
        return None
    text = str(raw)
    return text if text and text != "None" else None
