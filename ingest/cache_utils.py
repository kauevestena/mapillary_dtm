# Cache utilities shared across Mapillary ingestion modules.
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

from .. import constants

log = logging.getLogger(__name__)


def _resolve_root(base: Path | str | None) -> Path:
    if base is None:
        return Path(constants.MAPILLARY_CACHE_ROOT)
    return Path(base)


def metadata_cache_dir(base: Path | str | None = None) -> Path:
    path = _resolve_root(base) / "metadata"
    path.mkdir(parents=True, exist_ok=True)
    return path


def imagery_cache_dir(base: Path | str | None = None) -> Path:
    path = _resolve_root(base) / "imagery"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sequence_imagery_dir(seq_id: str, base: Path | str | None = None) -> Path:
    directory = imagery_cache_dir(base=base) / str(seq_id)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def directory_size_bytes(path: Path) -> int:
    total = 0
    for file_path in path.rglob("*"):
        try:
            if file_path.is_file():
                total += file_path.stat().st_size
        except OSError:
            continue
    return total


def enforce_quota(path: Path, max_gb: float) -> Tuple[int, List[Path]]:
    """Ensure directory *path* stays under max_gb by deleting oldest files."""
    if max_gb <= 0:
        return directory_size_bytes(path), []

    limit_bytes = int(max_gb * (1024**3))
    file_paths = [p for p in path.rglob("*") if p.is_file()]
    entries: List[Tuple[float, Path, int]] = []
    for file_path in file_paths:
        try:
            stat = file_path.stat()
        except OSError:
            continue
        entries.append((stat.st_mtime, file_path, stat.st_size))

    if not entries:
        return 0, []

    entries.sort(key=lambda item: item[0])
    total_bytes = sum(item[2] for item in entries)
    removed: List[Path] = []

    if total_bytes <= limit_bytes:
        return total_bytes, removed

    for _, file_path, size in entries:
        try:
            file_path.unlink()
            removed.append(file_path)
            total_bytes -= size
        except OSError as exc:
            log.warning("Failed to remove cache file %s: %s", file_path, exc)
        if total_bytes <= limit_bytes:
            break

    if removed:
        log.info(
            "Pruned %d cache artefacts from %s to enforce %.2f GB quota",
            len(removed),
            path,
            max_gb,
        )

    return total_bytes, removed
