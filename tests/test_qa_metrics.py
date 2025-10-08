import numpy as np

from dtm_from_mapillary.cli.pipeline import _summarize_agreement
from dtm_from_mapillary.qa.qa_internal import slope_from_plane_fit, write_agreement_maps


def almost_equal(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def test_slope_from_plane_fit_planar_surface():
    res = 0.5
    size = 41
    coords = np.arange(size) * res
    xx, yy = np.meshgrid(coords, coords)
    plane = 2.0 * xx + 0.5 * yy + 3.0

    slope, aspect = slope_from_plane_fit(plane, win=5)
    mask = np.isfinite(slope)
    expected_slope = np.degrees(np.arctan(np.hypot(2.0, 0.5)))

    mean_slope = float(np.nanmean(slope[mask]))
    mean_aspect = float(np.nanmean(aspect[mask]))

    assert abs(mean_slope - expected_slope) < 2.0
    assert abs(mean_aspect - 284.0) < 5.0


def test_write_agreement_maps_basic_stats():
    fused = np.zeros((3, 3), dtype=np.float32)
    sources = {
        "opensfm": fused + 0.2,
        "colmap": fused - 0.3,
    }
    results = write_agreement_maps(None, fused, sources)

    assert set(results.keys()) == {
        "dz_mean_abs",
        "dz_rmse",
        "dz_max_abs",
        "slope_mean_abs",
        "source_count",
    }
    mean_abs = float(np.nanmean(results["dz_mean_abs"]))
    max_abs = float(np.nanmax(results["dz_max_abs"]))
    assert almost_equal(mean_abs, 0.25, tol=1e-3)
    assert almost_equal(max_abs, 0.3, tol=1e-6)
    assert np.all(np.nan_to_num(results["source_count"]) >= 0.0)


def test_summarize_agreement_reduces_arrays():
    diff_map = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=float)
    summary = _summarize_agreement({"dz_mean_abs": diff_map})
    assert "dz_mean_abs" in summary
    stats = summary["dz_mean_abs"]
    assert almost_equal(stats["mean"], 2.0)
    assert abs(stats["p95"] - 3.0) < 0.2
