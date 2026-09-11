"""Numerical and evidence regressions using the tracked, real SfM dataset.

No fabricated points or model outputs. Coordinate transformations of real
points test invariances; these tests do not validate street-level slope accuracy.
"""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial import cKDTree

from slope_from_mapillary.cli import prepare_project
from slope_from_mapillary.estimation import (FitPolicy, angular_bound, directional_allowance,
    directional_grades, fit_plane, screening_status)
from slope_from_mapillary.grid import Grid, Measurement, fuse_cells
from slope_from_mapillary.pipeline import audit_dataset, audit_project, run_pipeline
from slope_from_mapillary.reconstruction import load_colmap, load_opensfm
from slope_from_mapillary.reference import register_horizontal, vertical_basis
from slope_from_mapillary.surfaces import ObservationPolicy, select_surfaces

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "qa/data/sample_dataset"
SFM = DATA / "geometry/opensfm/l27kwlcx3fjh7t6w9ccvic"
COLMAP = DATA / "geometry/colmap/l27kwlcx3fjh7t6w9ccvic/sparse_txt/0"


class RealReconstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_opensfm(SFM / "reconstruction.json", SFM / "tracks.csv")
        cls.points = np.array([point.xyz for point in cls.model.points.values()])
        # Raw reconstruction units are uncalibrated. Use a numerical tolerance
        # based on their measured spread ONLY for transformation regressions.
        # The production default policy is separately tested to reject them.
        _, indices = cKDTree(cls.points).query(cls.points[len(cls.points)//2], k=32)
        cls.local = cls.points[indices]
        cls.center = cls.local[:, :2].mean(axis=0)
        span = np.ptp(cls.local, axis=0)
        cls.numerical_policy = FitPolicy(min_span_m=float(min(span[:2])/10),
            max_gap_m=float(np.linalg.norm(span)), max_residual_m=float(span[2]*2),
            max_fit_precision_deg=89, min_inlier_fraction=1, max_slope_deg=89)
        cls.plane, reason = fit_plane(cls.local, cls.center, cls.numerical_policy)
        if cls.plane is None:
            raise AssertionError(f"Real numerical fixture could not be fit: {reason}")

    def test_load_real_opensfm_tracks_and_pose_convention(self):
        source = json.loads((SFM / "reconstruction.json").read_text())[0]
        self.assertEqual(len(self.model.points), len(source["points"]))
        self.assertTrue(all(point.observations for point in self.model.points.values()))
        for name, shot in self.model.shots.items():
            np.testing.assert_allclose(shot.rotation_cw @ shot.center + source["shots"][name]["translation"],
                                       0, atol=1e-10)
            for point in self.model.points.values():
                if name in point.observations:
                    x, y = point.observations[name]
                    self.assertGreaterEqual(x, 0)
                    self.assertLess(x, shot.width)
                    self.assertGreaterEqual(y, 0)
                    self.assertLess(y, shot.height)

    def test_load_real_colmap_tracks(self):
        model = load_colmap(COLMAP)
        self.assertGreater(len(model.points), 0)
        self.assertTrue(all(len(point.observations) >= 2 for point in model.points.values()))
        self.assertTrue(all(np.isfinite(point.error_px) for point in model.points.values()))

    def test_retained_backend_runners_share_production_readers_and_raw_frame(self):
        from dtm_from_mapillary.common_core import FrameMeta
        from dtm_from_mapillary.geom.opensfm_adapter import OpenSfMRunner
        from dtm_from_mapillary.geom.colmap_adapter import COLMAPRunner
        metadata = json.loads((DATA/"metadata.json").read_text())
        sequences = {key: [FrameMeta.from_dict(row) for row in rows] for key, rows in metadata.items()}
        with tempfile.TemporaryDirectory() as directory:
            a = OpenSfMRunner(workspace_root=directory)._load_fixture(SFM/"reconstruction.json", sequences)
            b = COLMAPRunner(workspace_root=directory)._load_fixture(COLMAP, sequences)
        self.assertTrue(a)
        self.assertTrue(b)
        for results in (a,b):
            self.assertTrue(all(result.coordinate_frame == "reconstruction" for result in results.values()))
        self.assertEqual(sum(len(result.points_xyz) for result in a.values()),len(self.model.points))

    def test_unobserved_points_are_not_given_views(self):
        model = load_opensfm(SFM / "reconstruction.json")
        self.assertTrue(all(not point.observations for point in model.points.values()))

    def test_sample_is_not_silently_accepted_as_calibrated(self):
        report = audit_dataset(DATA)
        self.assertEqual(report["status"], "needs_slope_evidence")
        self.assertEqual({row["format"] for row in report["reconstructions"]}, {"opensfm", "colmap"})

    def test_missing_semantics_never_become_ground(self):
        with tempfile.TemporaryDirectory() as directory:
            cloud = select_surfaces(self.model, Path(directory),
                {"1": {"id": "road", "kind": "road"}}, ObservationPolicy())
        self.assertEqual(len(cloud.xyz), 0)
        self.assertEqual(cloud.stats["available_masks"], 0)
        self.assertEqual(sum(cloud.stats["rejected"].values()), len(self.model.points))

    def test_prepare_audit_and_run_fail_for_the_actual_missing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"new/nested/project.json"
            prepare_project(SFM/"reconstruction.json", "opensfm", path,
                            tracks=SFM/"tracks.csv", metadata=DATA/"metadata.json")
            config = json.loads(path.read_text())
            self.assertIsNone(config["reconstructions"][0]["vertical_reference"])
            controls = config["reconstructions"][0]["georeference"]["controls"]
            self.assertTrue(controls)
            self.assertTrue(all("altitude" not in row and "alt_ellip" not in row for row in controls))
            report = audit_project(path)
            self.assertEqual(report["status"], "blocked")
            with self.assertRaisesRegex(ValueError, "evidence is missing"):
                run_pipeline(path, Path(directory)/"out")
            self.assertFalse(list((Path(directory)/"out").rglob("*.tif")))
            self.assertTrue((Path(directory)/"out/audit.json").exists())

    def test_vertical_reference_cannot_be_a_gps_or_camera_pitch_claim(self):
        for reference in (None, {"source": "gps_altitudes", "up": [0,0,1], "evidence": "GPS"},
                          {"source": "camera_pitch", "up": [0,0,1], "evidence": "pitch"}):
            with self.assertRaises(ValueError):
                vertical_basis(reference)

    def test_default_fit_rejects_unfiltered_scene(self):
        plane, _ = fit_plane(self.local, self.center, FitPolicy())
        self.assertIsNone(plane)

    def test_arbitrary_height_offset_and_map_translation_do_not_change_slope(self):
        shift = self.points[0] * 10_000  # translation derived from real coordinates
        transformed, reason = fit_plane(self.local + shift, self.center + shift[:2], self.numerical_policy)
        self.assertEqual(reason, "accepted")
        np.testing.assert_allclose(transformed.gradient, self.plane.gradient, rtol=1e-7, atol=1e-8)

    def test_uniform_scale_does_not_change_slope(self):
        factor = np.linalg.norm(self.model.shots[next(iter(self.model.shots))].center)
        policy = replace(self.numerical_policy, min_span_m=self.numerical_policy.min_span_m*factor,
                         max_gap_m=self.numerical_policy.max_gap_m*factor,
                         max_residual_m=self.numerical_policy.max_residual_m*factor)
        transformed, reason = fit_plane(self.local*factor, self.center*factor, policy)
        self.assertEqual(reason, "accepted")
        np.testing.assert_allclose(transformed.gradient, self.plane.gradient, atol=1e-10)

    def test_slope_units_and_reverse_direction(self):
        tangent = self.local[-1,:2] - self.local[0,:2]
        forward = directional_grades(self.plane.gradient, tangent)
        reverse = directional_grades(self.plane.gradient, -tangent)
        np.testing.assert_allclose(forward, -np.array(reverse), atol=1e-10)
        self.assertAlmostEqual(np.hypot(*forward), self.plane.slope_pct)
        self.assertAlmostEqual(100*np.tan(np.radians(self.plane.slope_deg)), self.plane.slope_pct)

    def test_directional_projection_does_not_mutate_path_vector(self):
        tangent = self.local[-1,:2] - self.local[0,:2]
        original = tangent.copy()
        bound = self.plane.fit_precision_deg
        projected = directional_allowance(self.plane.gradient, tangent, bound)
        self.assertGreaterEqual(projected, bound - 1e-10)
        np.testing.assert_array_equal(tangent, original)

    def test_unknown_accuracy_is_not_replaced_with_fit_precision(self):
        self.assertIsNone(angular_bound(self.plane, None, None))
        self.assertEqual(screening_status(self.plane.slope_pct, None, self.plane.slope_pct), "uncertainty_unknown")
        self.assertEqual(screening_status(self.plane.slope_pct, self.plane.fit_precision_deg,
                                          self.plane.slope_pct), "uncertain")

    def test_fusion_does_not_count_shared_images_as_independent(self):
        measurement = Measurement(self.plane, "opensfm", "real-sample", None, 0,
                                  list(self.model.points)[:32])
        cells, conflicts = fuse_cells({(0,0): [measurement, measurement]}, 2)
        self.assertEqual(conflicts, 0)
        self.assertEqual(cells[(0,0)].independent_groups, 1)
        self.assertIsNone(cells[(0,0)].bound_deg)

    def test_footprint_outside_observations_is_rejected(self):
        far = self.center + 10*np.ptp(self.local[:,:2], axis=0)
        plane, _ = fit_plane(self.local, far, self.numerical_policy)
        self.assertIsNone(plane)

    def test_grid_size_limit(self):
        with self.assertRaisesRegex(ValueError, "limit"):
            Grid.from_points(self.points, 0.5, max_cells=1)

    def test_invalid_policy_rejected(self):
        for kwargs in ({"max_gap_m": float("nan")}, {"min_points": 2}, {"max_residual_m": -1}):
            with self.assertRaises(ValueError):
                FitPolicy(**kwargs)
        with self.assertRaises(ValueError):
            ObservationPolicy(min_views=1)


if __name__ == "__main__":
    unittest.main()
