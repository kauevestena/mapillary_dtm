from pathlib import Path

p = Path("ARCHITECTURE.md")
if p.exists():
    content = p.read_text()
    
    insertion = """
## Core Pipeline Goal: Reducing to Terrain Level

The fundamental goal of the Mapillary DTM pipeline is to accurately reduce the 3D point of the camera perspective center down to the **road level** (terrain). This objective mandates that we strictly process frames with usable terrain geometry. 

During ingestion and pre-processing:
- We rely on HuggingFace `OneFormer` segmentations to extract exact **road** semantics.
- For feature matching, SfM, and VO, the pipeline uses the *entire* image context to maximize geometric stability and robust camera poses.
- However, for the final 3D reconstruction and depth mapping, **only the road masks** are utilized.
- Crucially, if an image contains absolutely no road mask (less than `MIN_ROAD_MASK_RATIO`), the image and its metadata are **completely removed** early in the pipeline to prevent useless processing.

"""
    # Just insert it after the main intro
    if "## System Architecture Overview" in content:
        content = content.replace("## System Architecture Overview", insertion + "## System Architecture Overview")
        p.write_text(content)
