"""Optional real model inference producing separate surface labels."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
from PIL import Image

from .pipeline import read_project
from .reconstruction import load


def segment_project(project: Path, images: Path, *, model_id: str, device: str = "cpu",
                    local_files_only: bool = False):
    """Run semantic segmentation on the exact SfM input images.

    Automatic labels distinguish road/sidewalk/terrain, not bridge levels or
    sidewalk instances. If there are multiple surfaces of a kind, edit labels
    to identify the physical surfaces before running the measurement pipeline.
    """
    import torch
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    project, images = Path(project).resolve(), Path(images).resolve()
    config = read_project(project)
    kinds = {}
    for code, surface in config["surfaces"].items():
        if surface["kind"] in kinds:
            raise ValueError("Multiple instances of a surface kind require reviewed instance masks")
        kinds[surface["kind"]] = int(code)
    processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=local_files_only)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_id, local_files_only=local_files_only)
    model.to(device).eval()
    id2label = {int(key): value.lower() for key, value in model.config.id2label.items()}
    if not any(label in kinds for label in id2label.values()):
        raise ValueError("The model labels do not match the project's surface kinds")
    files = {}
    for path in images.rglob("*"):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            files.setdefault(path.name, []).append(path)
    written = []
    for spec in config["reconstructions"]:
        reconstruction = load(spec, project.parent)
        destination = project.parent / spec["masks"]
        destination.mkdir(parents=True, exist_ok=True)
        for name, shot in reconstruction.shots.items():
            basename = Path(name).name
            candidates = files.get(basename, [])
            if not candidates:
                candidates = files.get(Path(basename).stem + "_1024.jpg", [])
            if not candidates:
                image_id = Path(basename).stem.rsplit("_", 1)[-1]
                if image_id.isdigit():
                    candidates = files.get(image_id + "_1024.jpg", [])
            if len(candidates) != 1:
                raise ValueError(f"Expected exactly one source image for {name}; found {len(candidates)}")
            with Image.open(candidates[0]) as image:
                image = image.convert("RGB")
                if not np.isclose(image.width / image.height, shot.width / shot.height, rtol=0.01):
                    raise ValueError(f"Source image aspect differs from reconstructed image: {name}")
                inputs = processor(images=image, return_tensors="pt").to(device)
                with torch.inference_mode():
                    logits = model(**inputs).logits
                    logits = torch.nn.functional.interpolate(logits, size=(image.height, image.width),
                                                             mode="bilinear", align_corners=False)
                    confidence, original = logits.softmax(dim=1).max(dim=1)
                confidence = confidence[0].float().cpu().numpy()
                original = original[0].cpu().numpy()
            labels = np.zeros(original.shape, dtype=np.uint16)
            for model_code, label in id2label.items():
                if label in kinds:
                    labels[original == model_code] = kinds[label]
            path = destination / (basename + ".npz")
            # Do not overwrite reviewed masks when rerunning inference.
            if path.exists():
                raise FileExistsError(f"Mask already exists; choose an empty mask directory: {path}")
            np.savez_compressed(path, labels=labels, confidence=confidence,
                                image_name=name, source=f"model:{model_id}",
                                model_revision=str(getattr(model.config, "_commit_hash", None)),
                                surface_mapping=json.dumps(config["surfaces"], sort_keys=True))
            written.append(str(path))
    return {"written_masks": len(written), "files": written,
            "review_needed": "Check surface boundaries, camera calibration, and separate stacked levels before slope estimation."}
