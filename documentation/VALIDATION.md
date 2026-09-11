# Validation and real sample findings

## What the tracked sample currently establishes

The authoritative `qa/data/sample_dataset` contains genuine Mapillary metadata,
images and cached reconstruction results. The new production readers import:

| Backend/component | Registered images | 3D points with actual tracks |
| --- | ---: | ---: |
| OpenSfM `l27kwlcx3fjh7t6w9ccvic`, component 0 | 12 | 4,071 |
| COLMAP `l27kwlcx3fjh7t6w9ccvic/sparse_txt/0` | 2 | 44 |

These numbers describe cached results, not fresh reconstruction runs. All OpenSfM
points have track observations in the tracked `tracks.csv`.

The bundle does **not** establish a calibrated vertical orientation, independent
slope checkpoints, or reviewed class/instance masks on the reconstructed images.
Its existing binary ground masks are for a different sequence. The COLMAP model
cannot satisfy the production default of three observed views. The OpenSfM model
uses a perspective camera while source metadata identifies fisheye captures;
that calibration choice needs review before interpreting fine surface geometry.

The old implementation was not solely taking GPS heights: it also ran image
reconstruction. However, camera-height-generated ground candidates, arbitrary
raw frames labeled ENU, and the elevation-first fusion/fill path could undermine
the independence and accuracy of its slope output. Those measurement paths have
been retired.

The sample project deliberately leaves `vertical_reference` null. Running its
audit or attempting production export is expected to stop on missing evidence.
Unfiltered sample scene points also fail the new default local support/planarity
gates in the numerical regression patch. No field-accurate Mapillary slope map
has been produced or claimed from this bundle.

## Software checks

Run `.venv/bin/python -m unittest discover -s tests/slope -v`.

- Core tests parse the real native models and actual tracks, check camera centers,
  reject absent semantics/calibration, and exercise prepare → audit → blocked run.
- Mathematical tests transform real reconstructed points to check translation and
  uniform-scale invariance, units, direction reversal, unknown uncertainty, and
  shared-image evidence grouping. The raw sample has uncalibrated units and noisy
  scene geometry; numerical tolerances derived from that real patch's spread are
  used only for these algebraic tests, never in the shipped measurement defaults.
- GIS tests use the tracked real `qa/data/qa_dtm.tif` to exercise the actual fitting,
  gridding and export functions, projected CRS checks, raster orientation/NoData,
  GeoJSON coordinates, exact grid-boundary path splitting and unknown path lengths.
  These are software regressions, not an independent Mapillary accuracy experiment.
- The opt-in inference test runs the actual SegFormer weights on the real registered
  COLMAP imagery and feeds the resulting masks to the production selector. The
  two-camera sample must still fail the three-view requirement.

The `Slope model validation` workflow installs the GIS dependencies on Python 3.11
and 3.12 and runs actual CPU inference in a separate job. Skipped optional local
checks do not count as successful execution; inspect the PR checks for execution
results. In the editing environment, NumPy/SciPy core tests were run. Local package
installation was unavailable, so GIS/model execution is delegated to those CI jobs.

## Field reference schema

`validate` consumes a WGS84 GeoJSON FeatureCollection of **Point** features. Each
feature has the following properties; all observations must be held out of fitting:

| Property | Meaning |
| --- | --- |
| `reference_id` | Unique identifier |
| `surface_id` | Exact physical surface ID from the model |
| `bearing_grid_deg` | Measurement direction clockwise from the project's projected grid north |
| `longitudinal_pct` | Signed percent rise in that direction |
| `cross_slope_pct` | Signed percent rise to that direction's right |
| `instrument` | Actual measurement instrument/procedure and calibration provenance |
| `measured_at` | Actual observation date/time |

Match the field measurement footprint to the modeled support window; a tiny
inclinometer reading and a four-metre fitted patch are different quantities on a
curved or irregular sidewalk. Use surveyed horizontal positions where GNSS error
could move the checkpoint across a curb or to another level. Do not silently
substitute true-north compass headings for grid bearings.

```bash
python -m slope_from_mapillary validate out/run-01 \
  --reference field_slopes.geojson --out out/run-01/field_comparison.json
```

The result retains unknown checkpoints in coverage counts and reports signed bias,
MAE, RMSE and the 95th percentile of absolute errors, in **percentage points**,
separately for longitudinal and cross slope and for each surface ID. A comparison
does not automatically mark every output cell as field-validated.

## Initial acceptance experiment

Collect overlapping sidewalk imagery, independently calibrated vertical attitude,
and held-out slopes across straight grades, cross slopes, ramps, crowns, curb
boundaries and occlusions. Include repeated passes and independent capture sessions.
Use appropriate camera models and test rolling-shutter/attitude sensitivity.

Choose required slope error and coverage targets from the intended accessibility
application before tuning. Report cross-slope errors separately: a method that is
adequate for broad terrain grades may still be inadequate for sidewalk cross slope.
Report registration error, unknown area/length, failure cases and threshold ambiguity
alongside accuracy. Avoid a precise-looking map whose missing observations were
silently filled.
