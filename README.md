# Surface slope models from Mapillary

This repository now estimates **observed surface slopes for pedestrian accessibility**.
The primary quantities are longitudinal slope, cross slope, and the local surface
normal. Absolute elevations and camera-height-derived DTMs are no longer products.
The repository URL remains `mapillary_dtm`; the Python package and command are
`slope_from_mapillary` and `mapillary-slope`.

**Status:** a research measurement pipeline with real SfM import, surface selection,
slope fitting, GIS exports, and field-comparison tools. The bundled imagery has
not established accessibility-grade slope accuracy. Its missing vertical and
semantic evidence is reported by the audit; no calibrated slope map is claimed.

## Why the redesign

A constant height offset does not change slope. Neither does a uniform 3D scale.
Consequently, local reconstructed surface geometry can support slope estimation
without accurate GNSS elevations. **Orientation relative to gravity still matters.**
Camera pitch, EXIF orientation, an arbitrary reconstruction Z axis, or fitting a
trajectory to noisy GPS altitudes cannot substitute for a calibrated vertical
reference. A one-degree orientation error is approximately 1.75 percentage points
of slope near horizontal.

The previous pipeline also manufactured some ground samples beneath the camera at
an assumed height. Those samples, the height solver, lower-envelope DTM fusion,
and corridor/TIN gap filling have been removed from the active code. The old
implementation remains in Git history.

## What it does

- Imports real OpenSfM reconstructions **with feature tracks**, or COLMAP text models.
- Uses a separately documented vertical reference; registers horizontally using
  camera controls, with one isotropic scale. GPS altitude never enters the slope path.
- Votes on surface labels at the **actual observed feature pixels**. Road, sidewalk,
  ramp, terrain, crossing, and different physical levels can have separate IDs.
- Fits supported local planes; rejects poor parallax, uncertain semantics, insufficient
  two-dimensional support, discontinuities, weak fits, and disagreements.
- Exports observed slope cells as GeoTIFF and WGS84 GeoJSON, with provenance and
  explicit unknowns. There is no terrain extrapolation or filling of unobserved blocks.
- Samples user-supplied paths for **signed longitudinal and rightward cross slope**,
  retaining unknown path lengths. Coordinate order defines travel direction.
- Evaluates exports against independent field slope measurements. Screening
  thresholds are explicit project settings, not automatic accessibility certification.

[Method and uncertainty](documentation/METHOD.md) ·
[Architecture](documentation/ARCHITECTURE.md) ·
[Validation and sample findings](documentation/VALIDATION.md) ·
[Migration](documentation/MIGRATION.md)

## Install

Python 3.10 or newer:

```bash
./setup_local.sh
source .venv/bin/activate
python -m slope_from_mapillary --help
```

Or `python -m pip install -e '.[dev]'`. The estimator does not require CUDA,
PyTorch, COLMAP, or OpenSfM executables when consuming existing reconstructions.
For actual semantic model inference, install `python -m pip install -e '.[segmentation]'`.
Model weights are downloaded only by the `segment` command; a local model path and
`--local-files-only` are supported. Consult the selected model's license.

The original Mapillary acquisition and optional OpenSfM/COLMAP/DIM runners remain
available from a repository checkout; install `requirements-reconstruction.txt`
(and `requirements-dim.txt` for DIM). They supply reconstructions to the new
measurement pipeline. Walking/cycling captures are no longer excluded by the
minimum-speed setting. Acquisition, binary installation, and new reconstruction
runs are separate from the lightweight estimator installation.

## Start with the real sample

```bash
python -m slope_from_mapillary audit-dataset qa/data/sample_dataset
python -m slope_from_mapillary audit examples/sample_project.json
```

The second command exits with status 2: **the sample lacks required calibration
and class-specific masks on its reconstructed images**. This is the expected
outcome, not a demo that pretends to produce validated slopes. It imports 4,071
OpenSfM points from 12 registered images, and the tracked COLMAP model contains
44 points from only two images.

To prepare your own project from an existing reconstruction:

```bash
python -m slope_from_mapillary prepare /path/to/reconstruction.json \
  --format opensfm --tracks /path/to/tracks.csv \
  --metadata /path/to/metadata.json --out slope_project.json
```

For COLMAP, give the directory containing `cameras.txt`, `images.txt`, and
`points3D.txt` and use `--format colmap`. Each disconnected component needs its
own project reconstruction record and vertical/registration evidence. The
`prepare` command fills horizontal controls from matching Mapillary metadata;
it leaves missing calibration as `null`.

1. Document the independently determined up vector and uncertainty in the raw
   reconstruction frame, following [METHOD.md](documentation/METHOD.md).
2. Generate actual surface masks, then review surface boundaries and levels:

   ```bash
   python -m slope_from_mapillary segment slope_project.json \
     --images /path/to/reconstructed/imagery --device cpu
   ```

3. Run the evidence audit, then the model:

   ```bash
   python -m slope_from_mapillary audit slope_project.json
   python -m slope_from_mapillary run slope_project.json --out-dir out/run-01
   ```

Use a new output directory for every run. A blocked run writes its audit and no
slope raster. An export is complete only when `manifest.json` says
`export_complete: true`. Empty/unsupported products carry `no_supported_cells`.

## Outputs

Each surface gets its own directory:

| File | Quantity |
| --- | --- |
| `slope_pct.tif`, `slope_deg.tif` | Maximum local slope magnitude |
| `gradient_east.tif`, `gradient_north.tif` | Directional gradient components, rise/run |
| `downslope_aspect_deg.tif` | Downslope bearing clockwise from grid north; undefined if flat |
| `angular_allowance_deg.tif` | Screening uncertainty budget; NoData if uncalibrated |
| `fit_precision_deg.tif` | Internal plane-fit precision; **not absolute accuracy** |
| `point_count.tif`, `independent_capture_groups.tif` | Evidence counts; same-image backends are grouped |
| `registration_rmse_m.tif` | Horizontal registration residual; relevant to narrow path matching |
| `patches.geojson`, `lineage.json` | Observed cells and links to contributing 3D point IDs |

The run also writes `audit.json`, `manifest.json`, and `report.html`. When the project
has `paths`, it writes `path_slopes.geojson` and path coverage summaries. All GeoJSON
uses longitude/latitude; rasters use the project's local projected metre CRS.
A 0.5 m pixel size is an output sampling choice, **not a claim of 0.5 m accuracy**.

To validate against held-out measurements:

```bash
python -m slope_from_mapillary validate out/run-01 \
  --reference field_slopes.geojson --out out/run-01/field_comparison.json
```

The reference schema and interpretation are in [VALIDATION.md](documentation/VALIDATION.md).

## Tests

```bash
.venv/bin/python -m unittest discover -s tests/slope -v
```

Tests consume the tracked real sample and actual core functions. Coordinate
transformations test mathematical invariance; they do not establish field accuracy.
GIS tests additionally use the tracked reference DTM. Optional inference tests run
real model weights on real imagery when `SLOPE_TEST_REAL_MODELS=1`.

Retain Mapillary imagery/contributor attribution and applicable source/model licenses.
