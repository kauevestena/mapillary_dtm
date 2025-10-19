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

_TOKEN_FILE_CANDIDATES = (
    "mapillary_token",
    ".secrets/mapillary_token",
    "config/mapillary_token",
)
_ENV_FILE_CANDIDATES = (".env", "config/runtime.env")


def _parse_token_line(line: str) -> Optional[str]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if "=" in stripped:
        key, value = stripped.split("=", 1)
        if key.strip() != "MAPILLARY_TOKEN":
            return None
        candidate = value.strip().strip('"').strip("'")
    else:
        candidate = stripped
    return candidate or None


def _read_token_from_path(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        for line in path.read_text(encoding="utf8").splitlines():
            candidate = _parse_token_line(line)
            if candidate:
                return candidate
    except OSError as exc:  # pragma: no cover - unlikely but worth logging
        log.warning("Failed to read token file %s: %s", path, exc)
    return None


def _read_token_from_env_files(repo_root: Path) -> Optional[str]:
    for rel_path in _ENV_FILE_CANDIDATES:
        token = _read_token_from_path(repo_root / rel_path)
        if token:
            return token
    return None


def _read_token_from_known_files(repo_root: Path) -> Optional[str]:
    for rel_path in _TOKEN_FILE_CANDIDATES:
        token = _read_token_from_path(repo_root / rel_path)
        if token:
            return token
    return None


def _resolve_token(explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit

    env_token = os.getenv("MAPILLARY_TOKEN")
    if env_token:
        return env_token

    token_file_env = os.getenv("MAPILLARY_TOKEN_FILE")
    if token_file_env:
        token = _read_token_from_path(Path(token_file_env).expanduser())
        if token:
            return token

    repo_root = Path(__file__).resolve().parents[2]
    token = _read_token_from_env_files(repo_root)
    if token:
        return token

    return _read_token_from_known_files(repo_root)


class MapillaryClient:
    """Thin convenience wrapper around the Mapillary Graph API."""

    _VECTOR_TILE_VERSION = 2

    def __init__(self, token: Optional[str] = None, timeout: int = 30, retries: int = 4):
        token = _resolve_token(token)
        if not token:
            raise RuntimeError(
                "Mapillary token not provided. Set MAPILLARY_TOKEN, provide MAPILLARY_TOKEN_FILE, "
                "place it in 'mapillary_token', or pass token=..."
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

    def get_thumbnail_url(self, image_id: str, resolution: int = constants.MAPILLARY_DEFAULT_IMAGE_RES) -> Optional[str]:
        field = f"thumb_{resolution}_url"
        data = self.get_image_meta(image_id, fields=[field])
        return data.get(field)

    def download_file(self, url: str, dest_path: Path, chunk_size: int = 1 << 20) -> None:
        """Download an asset pointed by *url* into *dest_path*."""
        with self.session.get(url, timeout=self.timeout, stream=True) as response:
            response.raise_for_status()
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with dest_path.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        fh.write(chunk)

    def download_thumbnail(self, image_id: str, dest_path: Path, resolution: int = constants.MAPILLARY_DEFAULT_IMAGE_RES, chunk_size: int = 1 << 20) -> Optional[Path]:
        """Download thumbnail for *image_id* if available."""
        url = self.get_thumbnail_url(image_id, resolution=resolution)
        if not url:
            return None
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        self.download_file(url, dest_path, chunk_size=chunk_size)
        return dest_path

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
        saw_tile_error = False
        for z, x, y in tiles:
            try:
                tile_bytes = self.get_vector_tile("sequences", z, x, y)
            except requests.HTTPError as exc:  # pragma: no cover - network errors hard to test
                log.warning("Failed to fetch vector tile %s/%s/%s: %s", z, x, y, exc)
                saw_tile_error = True
                continue
            extracted = self._extract_sequence_ids(tile_bytes)
            if extracted:
                seq_ids.update(extracted)
        if seq_ids:
            return seq_ids

        if saw_tile_error:
            log.info("Falling back to Mapillary Graph API search for sequences")
        fallback_ids = self._list_sequence_ids_via_graph_api(bbox)
        if fallback_ids:
            seq_ids.update(fallback_ids)
        return seq_ids

    def list_images_in_bbox(self, bbox: Sequence[float], limit: int = 2_000) -> List[Dict]:
        """Fetch image metadata inside *bbox* directly via Graph API."""
        lon_min, lat_min, lon_max, lat_max = map(float, bbox)
        params = {
            "bbox": f"{lon_min},{lat_min},{lon_max},{lat_max}",
            "fields": ",".join(
                {
                    "id",
                    "sequence_id",
                    "sequence{id}",
                    "geometry",
                    "captured_at",
                    "camera_type",
                    "camera_parameters",
                    "quality_score",
                    "thumb_1024_url",
                }
            ),
            "limit": min(limit, 200),
        }
        url = f"{constants.MAPILLARY_GRAPH_URL}/images"
        collected: List[Dict] = []
        while url and len(collected) < limit:
            try:
                resp = self._get(url, params=params)
            except requests.HTTPError as exc:
                log.warning("Graph API bbox image fetch failed for %s: %s", bbox, exc)
                break

            params = None  # only used on the first request
            payload = resp.json() if resp.content else {}
            data = payload.get("data") if isinstance(payload, dict) else None
            if isinstance(data, list):
                collected.extend(item for item in data if isinstance(item, dict))

            paging = payload.get("paging") if isinstance(payload, dict) else None
            if isinstance(paging, dict):
                next_url = paging.get("next")
                url = next_url if next_url else None
            else:
                break

        if len(collected) > limit:
            collected = collected[:limit]
        return collected

    def _list_sequence_ids_via_graph_api(self, bbox: Sequence[float]) -> Set[str]:
        """Fallback path: query Graph API directly for images in the bbox."""
        lon_min, lat_min, lon_max, lat_max = map(float, bbox)
        params = {
            "bbox": f"{lon_min},{lat_min},{lon_max},{lat_max}",
            "fields": "id,sequence{id}",
            "limit": 200,
        }
        url = f"{constants.MAPILLARY_GRAPH_URL}/images"
        seq_ids: Set[str] = set()

        while url:
            try:
                resp = self._get(url, params=params)
            except requests.HTTPError as exc:
                log.warning("Graph API fallback failed for bbox %s: %s", bbox, exc)
                break

            params = None  # only used on the first request
            payload = resp.json() if resp.content else {}
            data = payload.get("data") if isinstance(payload, dict) else None
            if isinstance(data, list):
                for item in data:
                    seq = None
                    if isinstance(item, dict):
                        seq = item.get("sequence") or item.get("sequence_id")
                        if isinstance(seq, dict):
                            seq = seq.get("id")
                    if seq:
                        seq_ids.add(str(seq))

            paging = payload.get("paging") if isinstance(payload, dict) else None
            if isinstance(paging, dict):
                next_url = paging.get("next")
                url = next_url if next_url else None
            else:
                break

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
