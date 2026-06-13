"""
External QA against held-out official GeoTIFFs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import rasterio
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, reproject

from ..io.readers import read_raster
from .qa_internal import slope_from_plane_fit


def compare_to_geotiff(
    dtm_path: str,
    check_path: str,
    *,
    out_dir: str | Path | None = None,
    reference_nodata_values: Sequence[float] | None = None,
) -> Dict[str, Any]:
    """Compare generated DTM against a reference GeoTIFF.

    The comparison is report-only: metric values never fail by threshold, but
    unreadable inputs, empty generated DTMs, or zero valid overlap raise.
    """

    dtm_arr, dtm_transform, dtm_crs = read_raster(dtm_path)
    if dtm_arr.ndim != 2:
        raise ValueError("DTM raster must be single-band")
    if dtm_crs is None:
        raise ValueError("DTM raster must have a CRS")

    dtm = np.asarray(dtm_arr, dtype=np.float32)
    generated_valid = np.isfinite(dtm)
    if not generated_valid.any():
        raise ValueError("Generated DTM has no finite cells")

    with rasterio.open(check_path) as ref:
        if ref.crs is None:
            raise ValueError("Reference DTM must have a CRS")
        ref_arr = ref.read(1).astype(np.float32, copy=False)
        ref_arr = _apply_reference_nodata(
            ref_arr,
            ref_nodata=ref.nodata,
            extra_nodata=reference_nodata_values,
        )
        ref_original_valid_count = int(np.isfinite(ref_arr).sum())
        if ref_original_valid_count == 0:
            raise ValueError("Reference DTM has no finite cells after nodata masking")

        ref_resampled = np.full(dtm.shape, np.nan, dtype=np.float32)
        reproject(
            source=ref_arr,
            destination=ref_resampled,
            src_transform=ref.transform,
            src_crs=ref.crs,
            src_nodata=np.nan,
            dst_transform=dtm_transform,
            dst_crs=dtm_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    reference_valid = np.isfinite(ref_resampled)
    mask = generated_valid & reference_valid
    n = int(mask.sum())
    if n == 0:
        raise ValueError("Generated DTM and reference DTM have zero valid overlap")

    dz_full = np.full(dtm.shape, np.nan, dtype=np.float32)
    dz_full[mask] = dtm[mask] - ref_resampled[mask]
    dz = dz_full[mask].astype(np.float64)
    abs_dz = np.abs(dz)

    slope_dtm, _ = slope_from_plane_fit(dtm)
    slope_ref, _ = slope_from_plane_fit(ref_resampled)
    slope_diff_full = np.full(dtm.shape, np.nan, dtype=np.float32)
    slope_mask = np.isfinite(slope_dtm) & np.isfinite(slope_ref) & mask
    if slope_mask.any():
        slope_diff_full[slope_mask] = slope_dtm[slope_mask] - slope_ref[slope_mask]
        slope_diff = slope_diff_full[slope_mask].astype(np.float64)
        rmse_slope = float(np.sqrt(np.mean(slope_diff**2)))
    else:
        rmse_slope = float("nan")

    stats: Dict[str, Any] = {
        "rmse_z": float(np.sqrt(np.mean(dz**2))),
        "bias_z": float(np.mean(dz)),
        "mae_z": float(np.mean(abs_dz)),
        "median_abs_z": float(np.median(abs_dz)),
        "p68_abs_z": float(np.percentile(abs_dz, 68)),
        "p95_abs_z": float(np.percentile(abs_dz, 95)),
        "std_z": float(np.std(dz)),
        "rmse_slope_deg": rmse_slope,
        "n": n,
        "slope_n": int(slope_mask.sum()),
        "generated_valid_count": int(generated_valid.sum()),
        "reference_valid_count": int(reference_valid.sum()),
        "reference_original_valid_count": ref_original_valid_count,
        "generated_valid_fraction": float(generated_valid.sum() / generated_valid.size),
        "reference_valid_fraction": float(reference_valid.sum() / reference_valid.size),
        "overlap_valid_fraction": float(n / generated_valid.size),
        "overlap_bounds": _mask_bounds(dtm_transform, mask),
        "generated_bounds": tuple(float(v) for v in array_bounds(*dtm.shape, dtm_transform)),
        "dtm_path": str(dtm_path),
        "reference_path": str(check_path),
    }
    hist_counts, hist_edges = np.histogram(abs_dz, bins=20)
    stats["abs_z_histogram"] = {
        "bin_edges": [float(v) for v in hist_edges],
        "counts": [int(v) for v in hist_counts],
    }

    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        artifacts = {
            "dz": str(out_path / "dz.tif"),
            "abs_dz": str(out_path / "abs_dz.tif"),
            "slope_diff_deg": str(out_path / "slope_diff_deg.tif"),
            "summary": str(out_path / "external_qa_summary.json"),
        }
        _write_float_raster(artifacts["dz"], dz_full, dtm_transform, dtm_crs)
        _write_float_raster(artifacts["abs_dz"], np.abs(dz_full), dtm_transform, dtm_crs)
        _write_float_raster(artifacts["slope_diff_deg"], slope_diff_full, dtm_transform, dtm_crs)
        (out_path / "external_qa_summary.json").write_text(
            json.dumps(stats, indent=2, default=_json_default),
            encoding="utf8",
        )
        stats["artifacts"] = artifacts

    return stats


def _apply_reference_nodata(
    ref_arr: np.ndarray,
    *,
    ref_nodata: float | int | None,
    extra_nodata: Sequence[float] | None,
) -> np.ndarray:
    arr = np.asarray(ref_arr, dtype=np.float32).copy()
    arr[~np.isfinite(arr)] = np.nan
    values: list[float] = []
    if ref_nodata is not None and np.isfinite(ref_nodata):
        values.append(float(ref_nodata))
    if extra_nodata:
        values.extend(float(v) for v in extra_nodata)
    for value in values:
        arr[np.isclose(arr, value, rtol=0.0, atol=1e-6)] = np.nan
    return arr


def _mask_bounds(transform, mask: np.ndarray) -> tuple[float, float, float, float] | None:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return None
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    x0, y0 = transform * (c0, r0)
    x1, y1 = transform * (c1, r1)
    return (float(min(x0, x1)), float(min(y0, y1)), float(max(x0, x1)), float(max(y0, y1)))


def _write_float_raster(path: str | Path, arr: np.ndarray, transform, crs) -> None:
    arr = np.asarray(arr, dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(arr, 1)


def _parse_nodata_values(raw: str | None) -> list[float]:
    if not raw:
        return []
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare a generated DTM to a reference GeoTIFF.")
    parser.add_argument("--generated-dtm", required=True)
    parser.add_argument("--reference-dtm", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--reference-nodata-values",
        default=None,
        help="Comma-separated reference values to treat as nodata, e.g. 0 or -9999,0.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    stats = compare_to_geotiff(
        args.generated_dtm,
        args.reference_dtm,
        out_dir=args.out_dir,
        reference_nodata_values=_parse_nodata_values(args.reference_nodata_values),
    )
    print(json.dumps(stats, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via CLI tests
    raise SystemExit(main())
