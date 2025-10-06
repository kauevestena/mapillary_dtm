"""
Lightweight Mapillary Graph API v4 + Vector Tiles client.
- Uses raw HTTP requests; you may replace with official SDKs if preferred.
- NOTE: Do not use `computed_*`, `sfm_cluster`, `mesh`, or `atomic_scale` for seeding—QA only.
"""
from __future__ import annotations
import os, time, logging, json, math
from typing import Dict, List, Iterable, Optional, Any, Tuple
import requests
from .. import constants

log = logging.getLogger(__name__)

class MapillaryClient:
    def __init__(self, token: Optional[str] = None, timeout: int = 30):
        self.token = token or os.getenv("MAPILLARY_TOKEN")
        if not self.token:
            raise RuntimeError("Mapillary token not provided. Set MAPILLARY_TOKEN or pass token=...")
        self.session = requests.Session()
        self.timeout = timeout

    # ------------- Graph API -------------
    def get_image_meta(self, image_id: str, fields: Optional[List[str]] = None) -> Dict:
        fields = fields or constants.DEFAULT_FIELDS
        url = f"{constants.MAPILLARY_GRAPH_URL}/{image_id}?fields={','.join(fields)}"
        headers = {"Authorization": f"OAuth {self.token}"}
        r = self.session.get(url, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def list_image_ids_in_sequence(self, seq_id: str, limit: int = 1000) -> List[str]:
        url = f"{constants.MAPILLARY_GRAPH_URL}/images"
        params = {
            "fields": "id,sequence_id",
            "sequence_id": seq_id,
            "limit": min(limit, 1000),
        }
        headers = {"Authorization": f"OAuth {self.token}"}
        ids = []
        while True:
            r = self.session.get(url, headers=headers, params=params, timeout=self.timeout)
            r.raise_for_status()
            js = r.json()
            ids += [x["id"] for x in js.get("data", [])]
            after = js.get("paging", {}).get("cursors", {}).get("after")
            if not after: break
            params["after"] = after
        return ids

    # ------------- Vector Tiles -------------
    def get_vector_tile(self, layer: str, z: int, x: int, y: int) -> bytes:
        """
        Fetch a vector tile for a given layer at z/x/y. Layers: 'images', 'sequences', 'map_feature', 'traffic_sign'.
        """
        url = f"{constants.MAPILLARY_TILES_URL}/{layer}/2/{z}/{x}/{y}"
        params = {"access_token": self.token}
        r = self.session.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.content
