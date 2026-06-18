import sys
from pathlib import Path
import os
import json
import numpy as np

try:
    from PIL import Image
except ImportError:
    print("Pillow required")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dtm_from_mapillary.semantics.ground_masks import prepare
from tests.sample_loader import get_sample_frames
from dtm_from_mapillary.constants import MIN_ROAD_MASK_RATIO
from dtm_from_mapillary.ingest.image_loader import ImageryLoader

def generate():
    dataset_dir = ROOT / "qa" / "data" / "sample_dataset"
    imagery_dir = dataset_dir / "imagery"
    masks_root = dataset_dir / "masks"
    
    bin_dir = masks_root / "binary"
    clip_dir = masks_root / "clipped"
    overlay_dir = masks_root / "overlay"
    
    bin_dir.mkdir(parents=True, exist_ok=True)
    clip_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    seqs, _ = get_sample_frames()
    
    print(f"Generating masks for sequences: {list(seqs.keys())}")
    
    # We force the backend to use the HF model
    # ensure default OneFormer model ID is configured properly
    os.environ["GROUND_MASK_MODEL_ID"] = "shi-labs/oneformer_cityscapes_swin_large"
    os.environ["DTM_MODELS_LOCAL_ONLY"] = "0"
    
    # Call prepare which will run OneFormer and save npz to bin_dir
    prepare(seqs, out_dir=bin_dir, imagery_root=dataset_dir, force=False, backend="model")
    
    loader = ImageryLoader(base=dataset_dir)
    
    for seq_id, frames in seqs.items():
        for frame in frames:
            npz_path = bin_dir / f"{frame.image_id}.npz"
            if not npz_path.exists():
                print(f"Missing {npz_path}")
                continue
                
            with np.load(npz_path) as data:
                prob = data["prob"]
            
            # Binary mask
            mask_bin = (prob > 0.5).astype(np.uint8)
            ratio = mask_bin.mean()
            print(f"Frame {frame.image_id} road ratio: {ratio*100:.1f}%")
            
            mask_png_path = bin_dir / f"{frame.image_id}.png"
            Image.fromarray(mask_bin * 255).save(mask_png_path)
            
            # Load original image
            img_arr = loader.load_rgb(frame)
            if img_arr is None:
                continue
                
            # If mask shape differs from image shape, resize mask using PIL
            h, w = img_arr.shape[:2]
            if (h, w) != mask_bin.shape:
                mask_pil = Image.fromarray(mask_bin * 255).resize((w, h), Image.Resampling.NEAREST)
                mask_bin = (np.array(mask_pil) > 127).astype(np.uint8)
            
            # Clipped image
            clipped = img_arr.copy()
            clipped[mask_bin == 0] = 0
            Image.fromarray(clipped).save(clip_dir / f"{frame.image_id}.jpg")
            
            # Overlay image
            # Create a red overlay
            overlay = img_arr.copy()
            red_layer = np.zeros_like(img_arr)
            red_layer[:, :, 0] = 255  # Red channel
            
            alpha = 0.5
            overlay_region = img_arr * (1 - alpha) + red_layer * alpha
            overlay[mask_bin == 1] = overlay_region[mask_bin == 1]
            
            Image.fromarray(overlay.astype(np.uint8)).save(overlay_dir / f"{frame.image_id}.jpg")
            
    print("Mask generation complete.")

if __name__ == "__main__":
    generate()
