"""
Adapter for running Deep-Image-Matching (DIM) feature extraction and matching.
DIM provides state-of-the-art learned features (SuperPoint, ALIKED) and matchers (LightGlue).
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

from .. import constants
from ..common_core import FrameMeta, ReconstructionResult
from .colmap_adapter import COLMAPConfig, COLMAPRunner, COLMAPUnavailable, _find_cached_image

log = logging.getLogger(__name__)

class DIMUnavailable(RuntimeError):
    pass

class DIMRunner:
    def __init__(
        self,
        workspace_root: Path | str,
        extractor: str = "superpoint",
        matcher: str = "lightglue",
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.extractor = extractor
        self.matcher = matcher
        
    def is_available(self) -> bool:
        try:
            import deep_image_matching
            return True
        except ImportError:
            return False

    def reconstruct(
        self,
        sequences: Mapping[str, Iterable[FrameMeta]],
        *,
        imagery_root: Optional[Path | str] = None,
        progress: bool = False,
    ) -> Dict[str, ReconstructionResult]:
        if not self.is_available():
            raise DIMUnavailable("deep_image_matching module not found.")

        # Let's use the COLMAPRunner to manage the workspace and mapping,
        # but we will use DIM for the feature extraction & matching steps.
        results: Dict[str, ReconstructionResult] = {}
        
        for seq_id, frames in sequences.items():
            frames_list = list(frames)
            if len(frames_list) < 2:
                continue
                
            seq_workspace = self.workspace_root / seq_id
            seq_workspace.mkdir(parents=True, exist_ok=True)
            
            # Move images_dir outside of seq_workspace so DIM config init doesn't delete it
            images_dir = self.workspace_root / f"{seq_id}_images"
            if images_dir.exists():
                shutil.rmtree(images_dir)
            images_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # 1. Stage images
                staged = 0
                sorted_frames = sorted(frames_list, key=lambda f: f.captured_at_ms)
                for frame in sorted_frames:
                    src = _find_cached_image(frame, imagery_root)
                    if src:
                        dest = images_dir / f"{staged:06d}_{frame.image_id}{src.suffix.lower()}"
                        if not dest.exists():
                            shutil.copy2(src, dest)
                        staged += 1
                        
                if staged < 2:
                    log.warning("DIM: Not enough images staged for seq %s", seq_id)
                    continue

                # 2. Run DIM extraction and matching
                db_path = seq_workspace / "database.db"
                if db_path.exists():
                    db_path.unlink()
                    
                try:
                    import deep_image_matching as dim
                    
                    if self.extractor == "no_extractor" or self.extractor is None or self.matcher in ["loftr", "se2loftr", "roma", "srif"]:
                        pipeline_name = self.matcher
                    else:
                        pipeline_name = f"{self.extractor}+{self.matcher}"
                    
                    # If we have less than 3 staged images, sequential matching strategy fails validation.
                    # Fallback to bruteforce/exhaustive matching in that case.
                    strategy = "sequential" if staged >= 3 else "bruteforce"
                    overlap = min(10, staged - 2) if (staged > 2 and strategy == "sequential") else 0
                    
                    args = {
                        "images": images_dir,
                        "outs": seq_workspace,
                        "pipeline": pipeline_name,
                        "strategy": strategy,
                        "overlap": overlap,
                        "quality": constants.DIM_QUALITY or "high",
                        "tiling": "none",
                        "force": True,
                    }
                    
                    # Setup DIM config
                    dim_cfg = dim.Config(args)
                    
                    # Initialize ImageMatcher class
                    matcher = dim.ImageMatcher(dim_cfg)
                    
                    # Run image matching and export to COLMAP format
                    feature_path, match_path = matcher.run()
                    
                    dim.io.export_to_colmap(
                        img_dir=images_dir,
                        feature_path=feature_path,
                        match_path=match_path,
                        database_path=str(db_path),
                        camera_config_path=dim_cfg.general["camera_options"],
                    )
                    
                except Exception as e:
                    log.error("DIM feature matching failed for seq %s: %s", seq_id, e)
                    continue
                
                # 3. Run COLMAP mapper on the DIM database
                colmap_runner = COLMAPRunner(workspace_root=seq_workspace)
                try:
                    # We skip extraction/matching and just run mapper
                    sparse_root = seq_workspace / "sparse"
                    if sparse_root.exists():
                        shutil.rmtree(sparse_root)
                    sparse_root.mkdir(parents=True, exist_ok=True)
                    
                    sparse_txt = seq_workspace / "sparse_txt"
                    if sparse_txt.exists():
                        shutil.rmtree(sparse_txt)
                    
                    colmap_runner._run_colmap(
                        [
                            "mapper",
                            "--database_path", str(db_path),
                            "--image_path", str(images_dir),
                            "--output_path", str(sparse_root),
                            "--Mapper.num_threads", str(constants.COLMAP_DEFAULT_THREADS),
                        ],
                        workspace=seq_workspace,
                        timeout=7200,
                    )
                    
                    # 4. Convert and load result
                    txt_dir = seq_workspace / "sparse_txt" / "0"
                    model_dir = sparse_root / "0"
                    if model_dir.exists():
                        colmap_runner._convert_model_to_text(model_dir, txt_dir)
                        seq_results = colmap_runner._load_fixture(txt_dir, {seq_id: frames_list})
                        if seq_id in seq_results:
                            res = seq_results[seq_id]
                            res.source = "dim"
                            results[seq_id] = res
                except Exception as e:
                    log.error("COLMAP mapping failed after DIM for seq %s: %s", seq_id, e)
            finally:
                if images_dir.exists():
                    shutil.rmtree(images_dir)
                
        return results
