"""
OSMnx helpers to derive corridor polygons (street vicinity).
"""
from __future__ import annotations

import logging
from typing import Mapping, Tuple

import math

from .. import constants

log = logging.getLogger(__name__)


def corridor_from_osm_bbox(
    bbox: Tuple[float, float, float, float],
    buffer_m: float | None = None,
    include_inner_blocks: bool | None = None,
) -> Mapping[str, object]:
    """Return corridor polygon(s) buffered around OSM road centerlines.

    Parameters
    ----------
    bbox:
        Bounding box as (lon_min, lat_min, lon_max, lat_max).
    buffer_m:
        Buffer distance in meters applied to road centerlines.
    include_inner_blocks:
        When ``True`` corridor polygons are returned without interior holes.
    """

    buffer_m = float(buffer_m if buffer_m is not None else constants.CORRIDOR_HALF_W_M)
    include_inner_blocks = (
        constants.INCLUDE_INNER_BLOCKS if include_inner_blocks is None else bool(include_inner_blocks)
    )

    bbox = tuple(float(v) for v in bbox)
    if len(bbox) != 4:
        raise ValueError("bbox must contain four values (lon_min, lat_min, lon_max, lat_max)")
    lon_min, lat_min, lon_max, lat_max = bbox
    if lon_min >= lon_max or lat_min >= lat_max:
        raise ValueError(f"Invalid bbox ordering: {bbox}")

    try:
        from shapely.geometry import LineString, MultiPolygon, Polygon, box
        from shapely.ops import unary_union
    except ImportError:
        log.warning("Shapely not available; using rectangular corridor approximation.")
        return _rectangle_fallback(bbox, buffer_m)

    corridors = None
    source = "fallback"
    base = box(lon_min, lat_min, lon_max, lat_max)

    try:
        import osmnx as ox

        log.debug("Fetching OSM network within bbox %s", bbox)
        G = ox.graph_from_bbox(lat_max, lat_min, lon_max, lon_min, network_type="drive", simplify=True)
        if G and G.number_of_edges() > 0:
            edges = ox.graph_to_gdfs(G, nodes=False, edges=True, fill_edge_geometry=True)
            if not edges.empty:
                geom = unary_union(edges.geometry.values)
                corridors = _buffer_and_prepare(geom, buffer_m, include_inner_blocks)
                source = "osmnx"
    except Exception as exc:  # pragma: no cover - network failures expected in CI
        log.warning("OSMnx corridor retrieval failed (%s). Falling back to bbox buffer.", exc)

    if corridors is None:
        corridors = _buffer_and_prepare(base.boundary if hasattr(base, "boundary") else base, buffer_m, include_inner_blocks)
    elif hasattr(corridors, "is_empty") and corridors.is_empty:
        log.debug("Using bbox fallback corridor")
        base = box(lon_min, lat_min, lon_max, lat_max)
        corridors = _buffer_and_prepare(base.boundary if hasattr(base, "boundary") else base, buffer_m, include_inner_blocks)

    return {
        "geometry": corridors,
        "crs": "EPSG:4326",
        "source": source,
        "buffer_m": buffer_m,
        "geometry_type": "shapely",
    }


def _buffer_and_prepare(geom, buffer_m: float, include_inner_blocks: bool):
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.ops import transform, unary_union

    geom_union = unary_union(geom)
    buffered = _buffer_in_meters(geom_union, buffer_m)
    buffered = buffered.buffer(0.0)

    if include_inner_blocks:
        buffered = _fill_holes(buffered)

    if isinstance(buffered, Polygon):
        return MultiPolygon([buffered])
    if isinstance(buffered, MultiPolygon):
        return buffered
    raise TypeError(f"Unexpected geometry type after buffering: {type(buffered)!r}")


def _buffer_in_meters(geom, distance_m: float):
    from shapely.ops import transform

    try:
        from pyproj import Transformer
    except ImportError as exc:  # pragma: no cover - requirements guarantee pyproj
        raise RuntimeError("pyproj is required for metric buffering") from exc

    forward = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
    backward = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform

    geom_m = transform(forward, geom)
    buffered_m = geom_m.buffer(distance_m)
    return transform(backward, buffered_m)


def _fill_holes(geom):
    from shapely.geometry import MultiPolygon, Polygon

    def _strip(p: Polygon) -> Polygon:
        return Polygon(p.exterior)

    if isinstance(geom, Polygon):
        return _strip(geom)
    if isinstance(geom, MultiPolygon):
        return MultiPolygon([_strip(poly) for poly in geom.geoms if not poly.is_empty])
    return geom


def _rectangle_fallback(bbox, buffer_m: float):
    lon_min, lat_min, lon_max, lat_max = bbox
    lat_mean = (lat_min + lat_max) * 0.5
    deg_lat = buffer_m / 111320.0
    deg_lon = buffer_m / (111320.0 * max(math.cos(math.radians(lat_mean)), 1e-6))
    rect = (
        lon_min - deg_lon,
        lat_min - deg_lat,
        lon_max + deg_lon,
        lat_max + deg_lat,
    )
    return {
        "geometry": {
            "type": "rectangle",
            "bbox": rect,
            "lat_mean": lat_mean,
            "buffer_m": buffer_m,
        },
        "crs": "EPSG:4326",
        "source": "fallback-rectangle",
        "buffer_m": buffer_m,
        "geometry_type": "rectangle",
    }
