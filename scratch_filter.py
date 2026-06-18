from pathlib import Path
p = Path("semantics/ground_masks.py")
content = p.read_text()

filter_code = """
def filter_sequences_by_mask(seqs: MutableMapping[str, List[FrameMeta]], mask_dir: Path | str) -> None:
    \"\"\"Filter out frames that have an insufficient road mask ratio.\"\"\"
    from ..constants import MIN_ROAD_MASK_RATIO
    mask_path_dir = Path(mask_dir)
    for seq_id, frames in list(seqs.items()):
        retained = []
        for frame in frames:
            mask_path = mask_path_dir / f"{frame.image_id}.npz"
            keep = True
            if mask_path.exists():
                try:
                    with np.load(mask_path) as data:
                        if "prob" in data:
                            prob = data["prob"]
                            ratio = float((prob > 0.5).mean())
                            if ratio < MIN_ROAD_MASK_RATIO:
                                log.info("Dropped frame %s due to insufficient road mask (%.1f%%)", frame.image_id, ratio * 100)
                                keep = False
                except Exception as exc:
                    log.debug("Failed to verify mask ratio for %s: %s", frame.image_id, exc)
            if keep:
                retained.append(frame)
        if not retained:
            log.info("Dropped entire sequence %s due to no road masks", seq_id)
            del seqs[seq_id]
        else:
            seqs[seq_id] = retained
"""

content += "\n" + filter_code
p.write_text(content)
