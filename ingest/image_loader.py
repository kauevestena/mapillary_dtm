"""
Helper utilities to load cached Mapillary imagery for downstream modules.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .. import constants
from ..common_core import FrameMeta
from . import cache_utils

try:  # Optional dependency
    import cv2  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    cv2 = None

log = logging.getLogger(__name__)


class ImageryLoader:
    def __init__(
        self,
        base: Optional[Path | str] = None,
        resolution: Optional[int] = None,
    ) -> None:
        self.base = Path(base) if base is not None else None
        self.resolution = resolution or constants.MAPILLARY_DEFAULT_IMAGE_RES

    def load_gray(self, frame: FrameMeta) -> Optional[np.ndarray]:
        image = self._read(frame)
        if image is None:
            return None
        if image.ndim == 2:
            return image
        if cv2 is None:  # pragma: no cover - cv2 missing
            return np.mean(image, axis=2).astype(np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def load_rgb(self, frame: FrameMeta) -> Optional[np.ndarray]:
        image = self._read(frame)
        if image is None:
            return None
        if image.ndim == 3 and image.shape[2] == 3:
            if cv2 is None:  # already BGR==RGB when cv2 missing
                return image
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.ndim == 2:
            return np.repeat(image[:, :, None], 3, axis=2)
        return image

    def _read(self, frame: FrameMeta) -> Optional[np.ndarray]:
        if cv2 is None:
            return None
        seq_dir = self._sequence_dir(frame.seq_id)
        for path in self._candidate_paths(seq_dir, frame.image_id):
            if not path.exists():
                continue
            try:
                image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            except Exception:
                continue
            if image is not None:
                return image
        return None

    def _sequence_dir(self, seq_id: str) -> Path:
        if self.base is not None:
            return Path(self.base) / str(seq_id)
        return cache_utils.sequence_imagery_dir(seq_id)

    def _candidate_paths(self, directory: Path, image_id: str) -> Iterable[Path]:
        res = self.resolution
        stem_variants = [
            f"{image_id}_{res}",
            image_id,
        ]
        suffixes = (".jpg", ".jpeg", ".png", ".bmp")
        for stem in stem_variants:
            for suffix in suffixes:
                yield directory / f"{stem}{suffix}"
