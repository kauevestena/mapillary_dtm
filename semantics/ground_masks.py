"""Utilities to produce (or load) per-image ground probability masks."""
from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

import numpy as np

from ..common_core import FrameMeta
from ..ingest.image_loader import ImageryLoader

try:  # Optional model-backed segmentation.
    import torch  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    torch = None

try:
    from ..gpu import detect_device, get_optimal_dtype, configure_torch_defaults
except Exception:  # pragma: no cover
    detect_device = None  # type: ignore[assignment]
    get_optimal_dtype = None  # type: ignore[assignment]
    configure_torch_defaults = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

DEFAULT_GROUND_MASK_MODEL_ID = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
DEFAULT_GROUND_MASK_LABELS = ("road", "sidewalk", "terrain")


def prepare(
    seqs: Mapping[str, List[FrameMeta]],
    out_dir: Path | str = Path("cache/masks"),
    backend: str = "soft-horizon",
    force: bool = False,
    *,
    model_path: Path | str | None = None,
    imagery_root: Path | str | None = None,
    progress: bool | None = None,
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
    model_path:
        Optional TorchScript model that returns a ground probability/logit map.
        If omitted, ``GROUND_MASK_MODEL_PATH`` is consulted.
    imagery_root:
        Optional root for cached imagery consumed by the model backend.
    allow_heuristic:
        When ``False``, missing cached/model masks raise instead of falling
        back to the horizon heuristic.
    progress:
        Show a tqdm progress bar when ``True``. When ``None``, tqdm is enabled
        only for interactive stderr streams.

    Returns
    -------
    dict
        Mapping of sequence id to list of mask file paths (aligned with the
        input frame order).
    """

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    require_provenance = True
    model = None
    model_initialized = False
    cached_count = 0
    generated_count = 0
    total_frames = sum(len(frames) for frames in seqs.values())

    summary: Dict[str, List[Path]] = {}
    with _progress_bar(total_frames, "Ground masks", progress) as pbar:
        for seq_id, frames in seqs.items():
            mask_paths: List[Path] = []
            for frame in frames:
                mask_path = out_path / f"{frame.image_id}.npz"
                prob = None
                provenance: dict[str, str | None] | None = None
                if not force:
                    cached = _load_existing_mask(
                        mask_path,
                        require_provenance=require_provenance,
                        load_array=False,
                    )
                    if cached is not None:
                        cached_count += 1
                        mask_paths.append(mask_path)
                        _progress_update(
                            pbar,
                            cached=cached_count,
                            generated=generated_count,
                        )
                        continue
                if not model_initialized:
                    model = _init_model_masker(model_path=model_path, imagery_root=imagery_root)
                    model_initialized = True
                if model is not None:
                    prob = model.predict(frame)
                    if prob is not None:
                        provenance = model.provenance()
                if prob is None:
                    raise RuntimeError(
                        "Ground mask missing. "
                        "Provide provenanced cached masks, set GROUND_MASK_MODEL_PATH, "
                        "or cache the configured Hugging Face ground model."
                    )

                _write_mask(mask_path, prob, frame, provenance=provenance)
                generated_count += 1
                mask_paths.append(mask_path)
                _progress_update(
                    pbar,
                    cached=cached_count,
                    generated=generated_count,
                )
            summary[seq_id] = mask_paths

    return summary


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


class _TorchMasker:
    def __init__(self, path: Path, imagery_root: Path | str | None) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is not available for ground mask model inference")
        self.loader = ImageryLoader(imagery_root)
        device_str = os.getenv("GROUND_MASK_DEVICE")
        if not device_str and detect_device is not None:
            device_str = detect_device()
        elif not device_str:
            device_str = "cpu"
        self.device = torch.device(device_str)
        self.dtype = get_optimal_dtype(device_str) if get_optimal_dtype else torch.float32
        self.model = torch.jit.load(str(path), map_location=self.device)
        self.model.eval()
        if self.dtype == torch.float16 and self.device.type == "cuda":
            self.model = self.model.half()
            log.info("TorchScript ground mask model: FP16 inference on %s", self.device)
        self.path = path

    def predict(self, frame: FrameMeta) -> Optional[np.ndarray]:
        image = self.loader.load_rgb(frame)
        if image is None:
            return None
        tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        with torch.inference_mode():
            output = self.model(tensor)
        if isinstance(output, (list, tuple)):
            output = output[0]
        arr = output.squeeze().detach().float().cpu().numpy().astype(np.float32, copy=False)
        if arr.ndim != 2:
            arr = arr.reshape(arr.shape[-2], arr.shape[-1])
        if float(np.nanmin(arr)) < 0.0 or float(np.nanmax(arr)) > 1.0:
            arr = 1.0 / (1.0 + np.exp(-arr))
        return np.clip(arr, 0.0, 1.0).astype(np.float32)

    def provenance(self) -> dict[str, str | None]:
        return {
            "source_type": "model",
            "backend": "torchscript",
            "model_id": str(self.path),
            "model_revision": None,
        }


class _TransformersSegFormerMasker:
    def __init__(self, model_id: str, imagery_root: Path | str | None) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is not available for SegFormer mask inference")
        try:
            from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
        except Exception as exc:  # pragma: no cover - depends on optional extras
            raise RuntimeError("transformers is not available for SegFormer masks") from exc

        self.loader = ImageryLoader(imagery_root)
        self.model_id = model_id
        self.revision = os.getenv("GROUND_MASK_MODEL_REVISION")
        self.cache_dir = os.getenv("DTM_MODEL_CACHE_DIR", "models/huggingface")
        device_str = os.getenv("GROUND_MASK_DEVICE")
        if not device_str and detect_device is not None:
            device_str = detect_device()
        elif not device_str:
            device_str = "cpu"
        self.device = torch.device(device_str)
        self.dtype = get_optimal_dtype(device_str) if get_optimal_dtype else torch.float32
        local_only = os.getenv("DTM_MODELS_LOCAL_ONLY", "1").lower() not in {"0", "false", "no"}
        self.processor = AutoImageProcessor.from_pretrained(
            model_id,
            revision=self.revision,
            cache_dir=self.cache_dir,
            local_files_only=local_only,
        )
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_id,
            revision=self.revision,
            cache_dir=self.cache_dir,
            local_files_only=local_only,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()
        if self.dtype == torch.float16 and self.device.type == "cuda":
            log.info("SegFormer ground mask model: FP16 inference on %s", self.device)

        # Apply global torch perf defaults
        if configure_torch_defaults is not None:
            configure_torch_defaults()

        raw_labels = os.getenv("GROUND_MASK_LABELS")
        wanted = tuple(
            item.strip().lower()
            for item in (raw_labels.split(",") if raw_labels else DEFAULT_GROUND_MASK_LABELS)
            if item.strip()
        )
        id2label = getattr(self.model.config, "id2label", {}) or {}
        self.ground_ids = [
            int(idx)
            for idx, label in id2label.items()
            if str(label).strip().lower() in wanted
        ]
        if not self.ground_ids:
            raise RuntimeError(
                f"Configured ground labels {wanted} were not found in {model_id}"
            )

    def predict(self, frame: FrameMeta) -> Optional[np.ndarray]:
        image = self.loader.load_rgb(frame)
        if image is None:
            return None
        try:
            from PIL import Image
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Pillow is required for SegFormer masks") from exc

        pil_image = Image.fromarray(image)
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {key: value.to(device=self.device, dtype=self.dtype if value.is_floating_point() else value.dtype) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[:, self.ground_ids, :, :].sum(dim=1)
            probs = torch.nn.functional.interpolate(
                probs.unsqueeze(1),
                size=(pil_image.height, pil_image.width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
        arr = probs.detach().float().cpu().numpy().astype(np.float32, copy=False)
        return np.clip(arr, 0.0, 1.0)

    def provenance(self) -> dict[str, str | None]:
        return {
            "source_type": "model",
            "backend": "transformers-segformer",
            "model_id": self.model_id,
            "model_revision": self.revision,
        }


def _init_model_masker(
    model_path: Path | str | None,
    imagery_root: Path | str | None,
) -> Optional[object]:
    raw_path = model_path or os.getenv("GROUND_MASK_MODEL_PATH")
    if raw_path:
        path = Path(raw_path)
        if not path.exists():
            log.warning("GROUND_MASK_MODEL_PATH=%s does not exist; model masks disabled", raw_path)
            return None
        try:
            return _TorchMasker(path, imagery_root=imagery_root)
        except Exception as exc:
            log.warning("Failed to initialize TorchScript ground mask model: %s", exc)
            return None

    model_id = os.getenv("GROUND_MASK_MODEL_ID", DEFAULT_GROUND_MASK_MODEL_ID)
    if not model_id:
        return None
    try:
        return _TransformersSegFormerMasker(model_id, imagery_root=imagery_root)
    except Exception as exc:
        log.warning("Failed to initialize SegFormer ground mask model: %s", exc)
        return None


def _should_init_model_masker(
    model_path: Path | str | None,
    *,
    allow_heuristic: bool,
) -> bool:
    if model_path or os.getenv("GROUND_MASK_MODEL_PATH"):
        return True
    if not allow_heuristic:
        return True
    return "GROUND_MASK_MODEL_ID" in os.environ and bool(os.getenv("GROUND_MASK_MODEL_ID"))


def _load_existing_mask(
    path: Path,
    *,
    require_provenance: bool = False,
    load_array: bool = True,
) -> Optional[tuple[np.ndarray | None, dict[str, str | None]]]:
    if not path.exists():
        return None
    try:
        with np.load(path) as data:
            if "prob" in data:
                arr = None
                if load_array:
                    arr = np.asarray(data["prob"], dtype=np.float32)
                    if arr.ndim != 2:
                        return None
                provenance = {
                    "source_type": _npz_scalar(data, "source_type"),
                    "backend": _npz_scalar(data, "backend"),
                    "model_id": _npz_scalar(data, "model_id"),
                    "model_revision": _npz_scalar(data, "model_revision"),
                }
                if require_provenance and provenance["source_type"] not in {"model", "external"}:
                    return None
                return arr, provenance
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


def _write_mask(
    path: Path,
    prob: np.ndarray,
    frame: FrameMeta,
    *,
    provenance: Mapping[str, str | None] | None = None,
) -> None:
    provenance = dict(provenance or {})
    try:
        np.savez_compressed(
            path,
            prob=np.clip(prob.astype(np.float32), 0.0, 1.0),
            image_id=frame.image_id,
            seq_id=frame.seq_id,
            captured_at_ms=frame.captured_at_ms,
            source_type=provenance.get("source_type") or "",
            backend=provenance.get("backend") or "",
            model_id=provenance.get("model_id") or "",
            model_revision=provenance.get("model_revision") or "",
        )
    except OSError as exc:  # pragma: no cover - filesystem error
        log.error("Failed to save mask %s: %s", path, exc)
        raise


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


def _safe_int(value) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
