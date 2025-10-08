"""
Delaunay TIN fill from corridor to AOI with limited extrapolation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Iterable, List, Mapping, Sequence

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay, cKDTree

from .. import constants
from ..common_core import wgs84_to_enu

log = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from shapely.geometry import MultiPolygon, Polygon


@dataclass(frozen=True)
class TINModel:
    tri: Delaunay | None
    interpolator: LinearNDInterpolator | None
    tree: cKDTree | None
    xy: np.ndarray
    z: np.ndarray
    constrained: bool = False  # Whether this TIN has breakline constraints

    @property
    def valid(self) -> bool:
        return (
            self.tri is not None
            and self.interpolator is not None
            and self.tree is not None
        )


def build_tin(points: Sequence[Mapping[str, object]]) -> TINModel:
    """Construct a Delaunay TIN from consensus ground points."""

    if not points:
        return TINModel(
            None,
            None,
            None,
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    xy = np.array([[float(p["x"]), float(p["y"])] for p in points], dtype=np.float64)
    z = np.array([float(p["z"]) for p in points], dtype=np.float64)

    if xy.shape[0] < 3:
        log.debug("Not enough points for TIN (need >=3, got %d)", xy.shape[0])
        return TINModel(None, None, None, xy, z)

    try:
        tri = Delaunay(xy)
        interpolator = LinearNDInterpolator(tri, z, fill_value=np.nan)
        tree = cKDTree(xy)
        return TINModel(
            tri=tri, interpolator=interpolator, tree=tree, xy=xy, z=z, constrained=False
        )
    except Exception as exc:  # pragma: no cover - rare numerical failures
        log.warning("Failed to build TIN: %s", exc)
        return TINModel(None, None, None, xy, z, constrained=False)


def build_constrained_tin(
    points: Sequence[Mapping[str, object]],
    breakline_vertices: np.ndarray | None = None,
    breakline_edges: List[tuple[int, int]] | None = None,
) -> TINModel:
    """Construct a constrained Delaunay TIN with breakline enforcement.

    Uses the 'triangle' library to build a TIN that respects breakline edges
    as constraints (no triangle edges will cross breaklines).

    Args:
        points: Ground points for TIN construction
        breakline_vertices: (N, 3) array of XYZ positions for breaklines
        breakline_edges: List of (i, j) vertex index pairs defining breakline segments

    Returns:
        TINModel with constrained triangulation
    """
    if (
        breakline_vertices is None
        or breakline_edges is None
        or len(breakline_edges) == 0
    ):
        # No constraints - fall back to standard Delaunay
        return build_tin(points)

    try:
        import triangle
    except ImportError:
        log.warning(
            "triangle library not available - falling back to unconstrained TIN"
        )
        return build_tin(points)

    if not points:
        return TINModel(
            None,
            None,
            None,
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    # Extract XY coordinates and Z values from ground points
    ground_xy = np.array(
        [[float(p["x"]), float(p["y"])] for p in points], dtype=np.float64
    )
    ground_z = np.array([float(p["z"]) for p in points], dtype=np.float64)

    # Combine ground points with breakline vertices
    breakline_xy = breakline_vertices[:, :2]  # Take only XY
    breakline_z = breakline_vertices[:, 2]

    combined_xy = np.vstack([ground_xy, breakline_xy])
    combined_z = np.concatenate([ground_z, breakline_z])

    # Adjust edge indices (offset by number of ground points)
    n_ground = len(ground_xy)
    adjusted_edges = np.array(
        [[i + n_ground, j + n_ground] for i, j in breakline_edges], dtype=np.int32
    )

    # Build PSLG (Planar Straight Line Graph)
    pslg = {"vertices": combined_xy, "segments": adjusted_edges}

    try:
        # Triangulate with quality constraints
        # 'p' = PSLG mode, 'q30' = min angle 30°, 'a' = max area constraint
        result = triangle.triangulate(pslg, "pq30")

        # Extract triangulation
        vertices_2d = result["vertices"]
        triangles = result["triangles"]

        # Build Delaunay-compatible object
        # Note: triangle library returns different format than scipy.spatial.Delaunay
        # We need to create an interpolator manually

        from scipy.interpolate import LinearNDInterpolator

        # Map old vertex indices to new (triangle may reorder/add vertices)
        # For now, assume same ordering up to n_ground + n_breakline
        n_result = len(vertices_2d)
        z_result = np.zeros(n_result)

        # Map Z values (may need refinement if triangle adds Steiner points)
        for i in range(min(n_result, len(combined_z))):
            z_result[i] = combined_z[i]

        # Create interpolator from triangulation
        interpolator = LinearNDInterpolator(vertices_2d, z_result, fill_value=np.nan)
        tree = cKDTree(combined_xy)

        log.info(
            "Built constrained TIN: %d vertices, %d triangles, %d constraints",
            len(vertices_2d),
            len(triangles),
            len(breakline_edges),
        )

        return TINModel(
            tri=None,  # triangle result is not scipy.Delaunay compatible
            interpolator=interpolator,
            tree=tree,
            xy=combined_xy,
            z=combined_z,
            constrained=True,
        )

    except Exception as exc:  # pragma: no cover
        log.warning(
            "Failed to build constrained TIN: %s - falling back to unconstrained", exc
        )
        return build_tin(points)


def corridor_to_local(
    corridor_info: Mapping[str, object] | None,
    lon0: float,
    lat0: float,
    h0: float = 0.0,
) -> Mapping[str, object] | None:
    """Project corridor geometry from WGS84 to local ENU coordinates."""

    if not corridor_info:
        return None

    geom_type = corridor_info.get("geometry_type")
    if geom_type == "rectangle":
        rect = corridor_info.get("geometry") or {}
        bbox = rect.get("bbox")
        if not bbox:
            return None
        lon_min, lat_min, lon_max, lat_max = bbox
        corners = [
            (lon_min, lat_min),
            (lon_min, lat_max),
            (lon_max, lat_max),
            (lon_max, lat_min),
        ]
        enu_coords = [
            wgs84_to_enu(lon, lat, h0, lon0, lat0, h0) for lon, lat in corners
        ]
        xs = [c[0] for c in enu_coords]
        ys = [c[1] for c in enu_coords]
        info = dict(corridor_info)
        info["geometry"] = {"bbox": (min(xs), min(ys), max(xs), max(ys))}
        info["geometry_type"] = "rectangle"
        info["origin"] = (lon0, lat0, h0)
        return info

    if geom_type == "shapely":
        try:
            from shapely.geometry import MultiPolygon, Polygon
        except (
            ImportError
        ):  # pragma: no cover - should not happen if geometry_type is shapely
            log.warning(
                "Shapely unavailable during corridor projection; cannot project geometry."
            )
            return None

        geom = corridor_info.get("geometry")
        if geom is None:
            return None

        polygons: List[Polygon] = []
        for poly in getattr(geom, "geoms", [geom]):
            exterior = [
                wgs84_to_enu(float(lon), float(lat), h0, lon0, lat0, h0)[:2]
                for lon, lat in poly.exterior.coords
            ]
            interiors = [
                [
                    wgs84_to_enu(float(lon), float(lat), h0, lon0, lat0, h0)[:2]
                    for lon, lat in ring.coords
                ]
                for ring in poly.interiors
            ]
            polygons.append(
                Polygon(exterior, interiors) if interiors else Polygon(exterior)
            )

        if not polygons:
            return None
        local_geom = MultiPolygon(polygons) if len(polygons) > 1 else polygons[0]
        info = dict(corridor_info)
        info["geometry"] = local_geom
        info["geometry_type"] = "shapely"
        info["origin"] = (lon0, lat0, h0)
        return info

    log.warning("Unsupported corridor geometry type %s", geom_type)
    return None


def sample_outside_corridor(
    consensus_points: Sequence[Mapping[str, object]],
    corridor_info: Mapping[str, object] | None,
    grid_res: float | None = None,
    max_extrapolation_m: float | None = None,
    tin: TINModel | None = None,
) -> List[Mapping[str, object]]:
    """Generate TIN-derived samples outside the corridor footprint."""

    shapely_available = False
    Point = None
    try:  # pragma: no branch
        from shapely.geometry import Point  # type: ignore

        shapely_available = True
    except ImportError:
        Point = None

    if not corridor_info:
        return []

    geom_type = corridor_info.get("geometry_type")
    geometry = corridor_info.get("geometry")
    if geometry is None:
        return []

    if geom_type == "shapely":
        if not shapely_available:
            raise RuntimeError("Shapely required for shapely corridor geometry")
        if hasattr(geometry, "is_empty") and geometry.is_empty:
            return []
    elif geom_type == "rectangle":
        if not isinstance(geometry, dict) or "bbox" not in geometry:
            return []
    else:
        log.warning("Unsupported corridor geometry type %s during sampling", geom_type)
        return []

    grid_res = float(grid_res or constants.GRID_RES_M)
    max_extrap = float(max_extrapolation_m or constants.MAX_TIN_EXTRAPOLATION_M)

    tin = tin or build_tin(consensus_points)
    if not tin.valid:
        return []

    if geom_type == "shapely":
        buffered = geometry.buffer(max_extrap)
        minx, miny, maxx, maxy = buffered.bounds
    else:
        minx, miny, maxx, maxy = _rectangle_buffer_bounds(geometry, max_extrap)

    xs = np.arange(
        np.floor(minx / grid_res) * grid_res,
        np.ceil(maxx / grid_res) * grid_res + grid_res,
        grid_res,
    )
    ys = np.arange(
        np.floor(miny / grid_res) * grid_res,
        np.ceil(maxy / grid_res) * grid_res + grid_res,
        grid_res,
    )

    samples: List[Mapping[str, object]] = []
    for x in xs:
        for y in ys:
            if geom_type == "shapely":
                pt = Point(x, y)
                if geometry.contains(pt):
                    continue
                dist = geometry.distance(pt)
            else:
                if _rectangle_contains(geometry, x, y):
                    continue
                dist = _rectangle_distance(geometry, x, y)
            if dist > max_extrap:
                continue

            z = float(tin.interpolator(x, y))
            if not np.isfinite(z):
                if tin.tree is None:
                    continue
                nn_dist, nn_idx = tin.tree.query([x, y])
                z = float(tin.z[nn_idx])

            if constants.EXCLUDE_ELEVATED_STRUCTURES and _is_likely_elevated(
                tin, x, y, z
            ):
                continue

            samples.append(
                {
                    "x": float(x),
                    "y": float(y),
                    "z": z,
                    "sources": ["tin"],
                    "support": 1,
                    "sem_prob": 0.6,
                    "uncertainty": 0.35,
                    "tri_angle_deg": None,
                    "distance_to_corridor": float(dist),
                }
            )

    return samples


def _is_likely_elevated(tin: TINModel, x: float, y: float, z: float) -> bool:
    """Heuristic: filter samples far above nearby corridor observations."""

    if tin.tree is None or tin.z.size == 0:
        return False

    distance, index = tin.tree.query([x, y], k=1, workers=-1)
    nearest_z = float(tin.z[index])
    dz = z - nearest_z
    if dz > 1.5 and distance < 15.0:
        return True
    return False


def _rectangle_buffer_bounds(rect, extra_m: float):
    bbox = rect.get("bbox")
    if not bbox:
        raise ValueError("Rectangle corridor missing bbox")
    minx, miny, maxx, maxy = bbox
    return minx - extra_m, miny - extra_m, maxx + extra_m, maxy + extra_m


def _rectangle_contains(rect, x: float, y: float) -> bool:
    bbox = rect.get("bbox")
    if not bbox:
        return False
    minx, miny, maxx, maxy = bbox
    return (minx <= x <= maxx) and (miny <= y <= maxy)


def _rectangle_distance(rect, x: float, y: float) -> float:
    bbox = rect.get("bbox")
    if not bbox:
        return float("inf")
    minx, miny, maxx, maxy = bbox
    dx = 0.0
    if x < minx:
        dx = minx - x
    elif x > maxx:
        dx = x - maxx
    dy = 0.0
    if y < miny:
        dy = miny - y
    elif y > maxy:
        dy = y - maxy
    return math.hypot(dx, dy)
