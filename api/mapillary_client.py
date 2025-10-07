"""Mapillary Graph API v4 + Vector Tiles helper client.

The client keeps the interface intentionally small:
- vector tile fetch for coverage discovery
- lightweight wrappers around `/images` Graph API endpoints

It avoids Mapillary `computed_*` or pre-computed SfM data; those are
reserved for QA only per project policy.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import requests
from requests.adapters import HTTPAdapter

try:  # urllib3 moved Retry in 2.x; keep compatibility with requests vendor copy
    from urllib3.util import Retry  # type: ignore
except ImportError:  # pragma: no cover - requests always vendors urllib3
    from urllib3.util.retry import Retry  # type: ignore

from mapbox_vector_tile import decode as decode_mvt

from .. import constants
from .tiles import bbox_to_z14_tiles

log = logging.getLogger(__name__)


def _read_token_from_file() -> Optional[str]:
    """Read token from `mapillary_token` file in repo root, if present."""
    repo_root = Path(__file__).resolve().parents[2]
    token_path = repo_root / "mapillary_token"
    if not token_path.exists():
        return None
    try:
        for line in token_path.read_text(encoding="utf8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return stripped
    except OSError as exc:  # pragma: no cover - unlikely but worth logging
        log.warning("Failed to read token file %s: %s", token_path, exc)
    return None


class MapillaryClient:
    """Thin convenience wrapper around the Mapillary Graph API."""

    _VECTOR_TILE_VERSION = 2

    def __init__(self, token: Optional[str] = None, timeout: int = 30, retries: int = 4):
        token = token or os.getenv("MAPILLARY_TOKEN") or _read_token_from_file()
        if not token:
            raise RuntimeError(
                "Mapillary token not provided. Set MAPILLARY_TOKEN, place it in "
                "'mapillary_token', or pass token=..."
            )

        self.token = token
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"OAuth {self.token}", "User-Agent": "dtm-from-mapillary/0.1"})

        retry = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=0.8,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods={"GET"},
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)

    # ------------------------------------------------------------------
    # Internal helpers
    def _get(self, url: str, *, params: Optional[Dict[str, str]] = None, stream: bool = False) -> requests.Response:
        resp = self.session.get(url, params=params, timeout=self.timeout, stream=stream)
        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", "1"))
            time.sleep(min(retry_after, 5.0))
        resp.raise_for_status()
        return resp

    # ------------------------------------------------------------------
    # Graph API wrappers
    def get_image_meta(self, image_id: str, fields: Optional[Sequence[str]] = None) -> Dict:
        fields = fields or constants.DEFAULT_FIELDS
        url = f"{constants.MAPILLARY_GRAPH_URL}/{image_id}"
        params = {"fields": ",".join(fields)}
        return self._get(url, params=params).json()

    def get_images_meta(self, image_ids: Sequence[str], fields: Optional[Sequence[str]] = None, chunk_size: int = 50) -> List[Dict]:
        if not image_ids:
            return []
        fields = fields or constants.DEFAULT_FIELDS
        url = f"{constants.MAPILLARY_GRAPH_URL}/images"
        collected: List[Dict] = []
        for start in range(0, len(image_ids), chunk_size):
            batch = image_ids[start : start + chunk_size]
            params = {"ids": ",".join(batch), "fields": ",".join(fields)}
            resp = self._get(url, params=params)
            data = resp.json().get("data", [])
            if not isinstance(data, list):
                log.warning("Unexpected response payload for ids %s", batch)
                continue
            collected.extend(data)
        return collected

    def list_image_ids_in_sequence(self, seq_id: str, limit: int = 10_000) -> List[str]:
        url = f"{constants.MAPILLARY_GRAPH_URL}/images"
        params = {
            "fields": "id,sequence_id",
            "sequence_id": seq_id,
            "limit": min(limit, 1000),
        }
        ids: List[str] = []
        while True:
            resp = self._get(url, params=params)
            payload = resp.json()
            ids.extend(str(x.get("id")) for x in payload.get("data", []) if x.get("id"))
            cursor = payload.get("paging", {}).get("cursors", {}).get("after")
            if not cursor or len(ids) >= limit:
                break
            params = dict(params)
            params["after"] = cursor
        if len(ids) > limit:
            ids = ids[:limit]
        return ids

    # ------------------------------------------------------------------
    # Vector tiles
    def get_vector_tile(self, layer: str, z: int, x: int, y: int) -> bytes:
        """Fetch raw vector tile bytes for a specific layer."""
        url = f"{constants.MAPILLARY_TILES_URL}/{layer}/{self._VECTOR_TILE_VERSION}/{z}/{x}/{y}"
        params = {"access_token": self.token}
        return self._get(url, params=params, stream=True).content

    def list_sequence_ids_in_bbox(self, bbox: Sequence[float]) -> Set[str]:
        seq_ids: Set[str] = set()
        try:
            tiles = bbox_to_z14_tiles(tuple(bbox))
        except Exception as exc:  # pragma: no cover - defensive, bbox validated upstream
            raise ValueError(f"Invalid bbox {bbox}: {exc}")
        for z, x, y in tiles:
            try:
                tile_bytes = self.get_vector_tile("sequences", z, x, y)
            except requests.HTTPError as exc:  # pragma: no cover - network errors hard to test
                log.warning("Failed to fetch vector tile %s/%s/%s: %s", z, x, y, exc)
                continue
            seq_ids.update(self._extract_sequence_ids(tile_bytes))
        return seq_ids

    @staticmethod
    def _extract_sequence_ids(tile_bytes: bytes) -> Set[str]:
        if not tile_bytes:
            return set()
        try:
            decoded = decode_mvt(tile_bytes)
        except Exception as exc:  # pragma: no cover - decode errors unexpected
            log.warning("Failed to decode vector tile: %s", exc)
            return set()
        layer = decoded.get("sequences")
        if not layer:
            return set()
        seq_ids: Set[str] = set()
        for feature in layer:
            props = feature.get("properties", {}) if isinstance(feature, dict) else {}
            seq_id = (
                props.get("id")
                or props.get("sequence_id")
                or feature.get("id") if isinstance(feature, dict) else None
            )
            if seq_id is None:
                continue
            seq_ids.add(str(seq_id))
        return seq_ids
