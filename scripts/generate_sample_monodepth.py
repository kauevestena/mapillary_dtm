#!/usr/风env python3
import sys
import os
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.sample_loader import get_sample_frames
from dtm_from_mapillary.depth import monodepth

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("generate_monodepth")

def main():
    log.info("Loading QA sample dataset...")
    seqs, imagery_root = get_sample_frames()
    
    depth_dir = ROOT / "qa" / "data" / "sample_dataset" / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)
    
    # We want to use HuggingFace Depth-Anything-V2 models. We need to allow it to download.
    os.environ["DTM_MODELS_LOCAL_ONLY"] = "0"
    
    log.info(f"Running Monodepth prediction using real HuggingFace model...")
    try:
        results = monodepth.predict_depths(
            seqs,
            out_dir=depth_dir,
            force=False, # reuse cache
            imagery_root=imagery_root,
            progress=True
        )
        log.info(f"Monodepth completed. Outputs serialized in {depth_dir}")
        
        # Validate and visualize
        vis_dir = depth_dir / "vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        for seq_id, frames in results.items():
            for image_id, result_dict in frames.items():
                depth = result_dict["depth"]
                uncert = result_dict["uncertainty"]
                log.info(f"Sample result for {image_id}: depth shape {depth.shape}, range [{depth.min():.2f}, {depth.max():.2f}]")
                
                # Save colorized depth map
                plt.imsave(str(vis_dir / f"{image_id}_depth.jpg"), depth, cmap='plasma')
                
                # Save colorized uncertainty map
                plt.imsave(str(vis_dir / f"{image_id}_uncert.jpg"), uncert, cmap='inferno')
                
        log.info(f"Visualizations saved to {vis_dir}")
            
    except Exception as e:
        log.error(f"Monodepth Track Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
