import os
from pathlib import Path

p = Path("semantics/ground_masks.py")
content = p.read_text()

# 1. Update DEFAULT_GROUND_MASK_MODEL_ID
content = content.replace(
    'DEFAULT_GROUND_MASK_MODEL_ID = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"',
    'DEFAULT_GROUND_MASK_MODEL_ID = "shi-labs/oneformer_cityscapes_dinat_large"'
)

# 2. Add _TransformersOneFormerMasker
oneformer_class = """

class _TransformersOneFormerMasker:
    def __init__(self, model_id: str, imagery_root: Path | str | None) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is not available for OneFormer mask inference")
        try:
            from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
        except Exception as exc:
            raise RuntimeError("transformers is not available for OneFormer masks") from exc

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
        
        self.processor = OneFormerProcessor.from_pretrained(
            model_id,
            revision=self.revision,
            cache_dir=self.cache_dir,
            local_files_only=local_only,
        )
        self.model = OneFormerForUniversalSegmentation.from_pretrained(
            model_id,
            revision=self.revision,
            cache_dir=self.cache_dir,
            local_files_only=local_only,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()
        if self.dtype == torch.float16 and self.device.type == "cuda":
            log.info("OneFormer ground mask model: FP16 inference on %s", self.device)

        if configure_torch_defaults is not None:
            configure_torch_defaults()

        raw_labels = os.getenv("GROUND_MASK_LABELS")
        wanted = tuple(
            item.strip().lower()
            for item in (raw_labels.split(",") if raw_labels else DEFAULT_GROUND_MASK_LABELS)
            if item.strip()
        )
        
        # Determine mapping from OneFormer class IDs to road masks
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
        except Exception as exc:
            raise RuntimeError("Pillow is required for OneFormer masks") from exc

        pil_image = Image.fromarray(image)
        # OneFormer processor requires 'task_inputs' for universal segmentation models
        inputs = self.processor(images=pil_image, task_inputs=["semantic"], return_tensors="pt")
        inputs = {key: value.to(device=self.device, dtype=self.dtype if value.is_floating_point() else value.dtype) for key, value in inputs.items()}
        
        with torch.inference_mode():
            outputs = self.model(**inputs)
            # Post-process to get semantic map
            predicted_map = self.processor.post_process_semantic_segmentation(outputs, target_sizes=[pil_image.size[::-1]])[0]
            
            # Map predictions to binary mask based on ground_ids
            mask = torch.zeros_like(predicted_map, dtype=torch.float32)
            for gid in self.ground_ids:
                mask[predicted_map == gid] = 1.0
                
        arr = mask.detach().float().cpu().numpy().astype(np.float32, copy=False)
        return np.clip(arr, 0.0, 1.0)

    def provenance(self) -> dict[str, str | None]:
        return {
            "source_type": "model",
            "backend": "transformers-oneformer",
            "model_id": self.model_id,
            "model_revision": self.revision,
        }
"""

content = content.replace("class _TransformersSegFormerMasker:", oneformer_class + "\n\nclass _TransformersSegFormerMasker:")

# 3. Update _init_model_masker to use OneFormer
old_init = """    try:
        return _TransformersSegFormerMasker(model_id, imagery_root=imagery_root)
    except Exception as exc:
        log.warning("Failed to initialize SegFormer ground mask model: %s", exc)
        return None"""
new_init = """    try:
        if "oneformer" in model_id.lower():
            return _TransformersOneFormerMasker(model_id, imagery_root=imagery_root)
        else:
            return _TransformersSegFormerMasker(model_id, imagery_root=imagery_root)
    except Exception as exc:
        log.warning("Failed to initialize ground mask model %s: %s", model_id, exc)
        return None"""

content = content.replace(old_init, new_init)

p.write_text(content)
