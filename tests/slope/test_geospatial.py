"""GIS regression using the tracked real reference DTM, never fabricated heights."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from slope_from_mapillary.estimation import FitPolicy, fit_plane
from slope_from_mapillary.grid import Cell, Grid, Measurement, estimate_cells, fuse_cells
from slope_from_mapillary.output import sample_paths, write_surface
from slope_from_mapillary.reference import metric_crs

ROOT = Path(__file__).resolve().parents[2]
HAS_GIS = all(importlib.util.find_spec(name) for name in ("rasterio", "pyproj"))


@unittest.skipUnless(HAS_GIS, "GIS dependencies unavailable; run the required CI GIS checks")
class RealRasterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import rasterio

        with rasterio.open(ROOT/"qa/data/qa_dtm.tif") as src:
            cls.crs = metric_crs(src.crs)
            values = src.read(1, masked=True)
            found = None
            for row in range(5, src.height-5, 9):
                for col in range(5, src.width-5, 9):
                    window = values[row-4:row+5, col-4:col+5]
                    if not np.ma.getmaskarray(window).any() and np.isfinite(window).all():
                        found = (row, col, window)
                        break
                if found:
                    break
            if found is None:
                raise AssertionError("Tracked reference DTM lacks a finite 9x9 window")
            row, col, window = found
            rr, cc = np.meshgrid(np.arange(row-4,row+5), np.arange(col-4,col+5), indexing="ij")
            x, y = rasterio.transform.xy(src.transform, rr.ravel(), cc.ravel())
            cls.points = np.column_stack([x, y, np.asarray(window).ravel()])
            cls.resolution = max(src.res)
        # These are GIS export tests, not claims about the reference survey's
        # accuracy. The real patch's observed spread sets numerical tolerances.
        cls.policy = FitPolicy(radius_m=cls.resolution*10, min_span_m=cls.resolution,
            max_gap_m=cls.resolution*2, max_residual_m=max(0.04,2*float(np.ptp(cls.points[:,2]))),
            max_slope_deg=89, max_fit_precision_deg=89)
        cls.grid = Grid.from_points(cls.points, cls.resolution, 10000)
        cls.key = (cls.grid.height//2, cls.grid.width//2)
        plane, reason = fit_plane(cls.points, cls.grid.center(*cls.key), cls.policy,
                                 cell_size_m=cls.grid.resolution)
        if plane is None:
            raise AssertionError(f"Real reference patch not exportable: {reason}")
        measurement = Measurement(plane, "reference-dtm", "reference-dtm", None, 0,
                                  [str(index) for index in range(len(cls.points))])
        cls.cell = Cell(plane.gradient, None, plane.fit_precision_deg, 1,
                        ["reference-dtm"], [measurement])

    def path(self, reverse=False):
        from pyproj import Transformer

        center = self.grid.center(*self.key)
        xy = np.array([[self.grid.west-self.resolution, center[1]],
                       [self.grid.west+(self.grid.width+1)*self.resolution, center[1]]])
        if reverse:
            xy = xy[::-1]
        project = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        lon, lat = project.transform(xy[:,0], xy[:,1])
        return {"type": "FeatureCollection", "features": [{"type": "Feature",
                "properties": {"surface_id": "reference-dtm"},
                "geometry": {"type": "LineString", "coordinates": np.column_stack([lon,lat]).tolist()}}]}

    def test_raster_transform_nodata_units_and_geojson(self):
        import rasterio
        from pyproj import Transformer

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            write_surface(out, "reference-dtm", {self.key:self.cell}, self.grid, self.crs, lineage=True)
            with rasterio.open(out/"gradient_north.tif") as src:
                self.assertEqual(src.crs, self.crs)
                self.assertTrue(np.isnan(src.nodata))
                self.assertAlmostEqual(float(src.read(1)[self.key]), self.cell.gradient[1], places=6)
                self.assertEqual(np.isfinite(src.read(1)).sum(), 1)
                np.testing.assert_allclose(src.xy(*self.key), self.grid.center(*self.key))
            with rasterio.open(out/"angular_allowance_deg.tif") as src:
                self.assertFalse(np.isfinite(src.read(1)).any())
            data = json.loads((out/"patches.geojson").read_text())
            ring = data["features"][0]["geometry"]["coordinates"][0]
            self.assertEqual(ring[0], ring[-1])
            self.assertTrue(all(abs(lon)<=180 and abs(lat)<=90 for lon,lat in ring))
            project = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
            polygon_xy = np.array([project.transform(*point) for point in ring[:-1]])
            np.testing.assert_allclose(polygon_xy.mean(axis=0), self.grid.center(*self.key), atol=1e-5)

    def test_path_splits_keep_unknown_length_and_reverse_signs(self):
        surfaces = {"reference-dtm": {self.key:self.cell}}
        threshold = {"longitudinal_pct":self.cell.observations[0].plane.slope_pct,
                     "cross_slope_pct":self.cell.observations[0].plane.slope_pct}
        forward, summary = sample_paths(self.path(), surfaces, self.grid, self.crs, threshold)
        backward, reverse_summary = sample_paths(self.path(True), surfaces, self.grid, self.crs, threshold)
        self.assertAlmostEqual(summary[0]["length_m"], (self.grid.width+2)*self.resolution, places=5)
        self.assertAlmostEqual(summary[0]["measured_length_m"], self.resolution, places=5)
        self.assertGreater(summary[0]["unknown_length_m"], 0)
        self.assertAlmostEqual(summary[0]["length_m"], reverse_summary[0]["length_m"], places=5)
        a = [f["properties"] for f in forward["features"] if f["properties"]["longitudinal_pct"] is not None]
        b = [f["properties"] for f in backward["features"] if f["properties"]["longitudinal_pct"] is not None]
        self.assertEqual(len(a), 1)
        self.assertEqual(len(b), 1)
        self.assertAlmostEqual(a[0]["longitudinal_pct"], -b[0]["longitudinal_pct"], places=5)
        self.assertAlmostEqual(a[0]["cross_slope_pct"], -b[0]["cross_slope_pct"], places=5)
        self.assertEqual(a[0]["longitudinal_screening"], "uncertainty_unknown")

    def test_missing_surface_is_not_substituted(self):
        path = self.path()
        path["features"][0]["properties"]["surface_id"] = "other-level"
        with self.assertRaisesRegex(ValueError, "surface_id"):
            sample_paths(path, {"reference-dtm":{self.key:self.cell}}, self.grid, self.crs, {})

    def test_observed_grid_and_full_normal_conflict(self):
        ids = [str(i) for i in range(len(self.points))]
        cells, reasons = estimate_cells(self.points, ids, self.grid, self.policy,
            source_id="reference-dtm", evidence_group="reference-dtm",
            vertical_uncertainty=None, geometry_uncertainty=None, registration_rmse_m=0)
        self.assertTrue(cells)
        self.assertGreater(reasons.get("accepted",0), 0)
        fused, conflicts = fuse_cells({key:[value,value] for key,value in cells.items()}, 2)
        self.assertEqual(conflicts,0)
        self.assertTrue(all(cell.independent_groups == 1 for cell in fused.values()))
        self.assertTrue(all(cell.bound_deg is None for cell in fused.values()))

    def test_metric_crs_rejects_degrees_webmercator_and_feet(self):
        for crs in ("EPSG:4326", "EPSG:3857", "EPSG:2263"):
            with self.assertRaises(ValueError):
                metric_crs(crs)


if __name__ == "__main__":
    unittest.main()
