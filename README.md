# DTM from Mapillary — High-Accuracy, Redundancy-Heavy Pipeline

**Goal:** Generate a **0.5 m** ellipsoidal-height **DTM (ground-only)** and **slope maps** from Mapillary imagery (car sequences only), maximizing **accuracy via redundancy** and **cross-validation**. No existing DTM is used except optionally as **initialization/QA**. The pipeline focuses on **slope fidelity** (accessibility mapping).

## Official Documentation
- **[documentation/index.html](documentation/index.html)** - Main official documentation landing page
- **[documentation/ARCHITECTURE.md](documentation/ARCHITECTURE.md)** - Architecture & Data Flow
- **[agents.md](agents.md)** - Guide for AI coding assistants

## Core Directives & Testing Philosophy

> **STRICT ENFORCEMENT:** No mock stuff allowed. All implementations must use real data, real model inferences, and authentic reconstruction paths.
> **PIPELINE INTEGRITY:** Don't implement parallel or ad-hoc scripts for testing. All validation, tests, and baby-steps MUST directly consume the core pipeline functions to ensure what is tested is exactly what is shipped.
> **SAMPLE DATASET:** The folder `qa/data/sample_dataset` is the authoritative fixture for all pipeline component testing.

## Attribution & Terms

Respect Mapillary's terms and attribution requirements. OSM data is © OpenStreetMap contributors.

**Highlights**
- Uses **two independent SfM stacks** (OpenSfM & COLMAP) + a **Deep-Image-Matching (DIM)** densifier for redundancy. (Legacy OpenCV VO is available via `--legacy-vo`).
- Fully **GPU-optimized** (CUDA 12.4 + FP16 + PyTorch batching) for Depth, Ground Masks, and COLMAP feature extraction.
- **Sequence-constant camera height** constraint \(h \in [1, 3]\,m\) + **GNSS deltas** + **semantic footpoint anchors** to resolve metric scale.
- **Ground-only** extraction by 3D semantic voting; **lower-envelope fusion** and **edge-aware smoothing** to preserve slope.
- **OSM-based corridor** via **OSMnx** to bound processing to street vicinity; **TIN fill** to extend from corridor into the full AOI with **max 5 m** extrapolation and always fill **inner blocks**.
- **Automatic elevated-structure masking** (bridges/overpasses) to avoid mixing deck elevations with terrain.
- Strict policy: Mapillary `computed_*`, `sfm_cluster`, `mesh`, `atomic_scale` are **QA-only** (never used for seeding/scale).

> **Heights:** ellipsoidal. Geoid correction can be applied later as optional post-processing.  
> **Corridor→AOI:** reconstruct inside corridor; fill outward using **Delaunay TIN**, limited extrapolation (`MAX_TIN_EXTRAPOLATION_M = 5`).

---

## Repository Layout

```
dtm_from_mapillary/
  constants.py
  common_core.py
  README.md
  requirements.txt
  .gitignore

  documentation/
    index.html
    ARCHITECTURE.md

  api/
    mapillary_client.py
    tiles.py

  io/
    geoutils.py
    readers.py
    writers.py

  tiling/
    tiler.py

  ingest/
    sequence_scan.py
    sequence_filter.py
    camera_models.py

  geom/
    sfm_opensfm.py
    sfm_colmap.py
    vo_simplified.py
    height_solver.py
    anchors.py

  semantics/
    ground_masks.py
    curb_edge_lane.py

  depth/
    monodepth.py
    plane_sweep_ground.py

  ground/
    ground_extract_3d.py
    recon_consensus.py
    corridor_fill_tin.py

  fusion/
    heightmap_fusion.py
    smoothing_regularization.py

  qa/
    qa_internal.py
    qa_external.py
    reports.py
    data/readme.md
    data/qa_dtm.tif
    data/qa_dtm_4326.tif
  osm/
    osmnx_utils.py

  cli/
    pipeline.py
```

---

## Quickstart

1. **Provision the environment**:
   - The easiest way is to use the provided setup script which will handle virtualenv creation, submodule init, and CUDA PyTorch installation:
   ```bash
   ./setup_local.sh
   source .venv/bin/activate
   ```
   - Alternatively, install manually:
   ```bash
   pip install -r requirements.txt
   ```
   - Optional extras:
     ```bash
     pip install -r requirements-optional.txt    # ML acceleration (PyTorch)
     pip install -r requirements-dim.txt         # Deep-Image-Matching support
     pip install -r requirements-dev.txt         # Developer tools + tests
     ```

2. **Cache production models and validate your setup**:
   ```bash
   python scripts/setup_production_models.py --accept-model-licenses
   ```
   > This downloads model weights into ignored local cache storage and writes `models/production_models.json`.

   ```bash
   python scripts/check_env.py --full --strict-production
   ```

3. **Set Mapillary API token** (Graph API v4):
   ```bash
   export MAPILLARY_TOKEN="YOUR_TOKEN_HERE"
   ```
   > Alternatively place the token in `.env` (as `MAPILLARY_TOKEN=...`) or in the repository `mapillary_token` file.

4. **Run the pipeline** over a bounding box (lon_min,lat_min,lon_max,lat_max):
   ```bash
   python -m dtm_from_mapillary.cli.pipeline run --aoi-bbox "-122.45,37.76,-122.41,37.79" --out-dir ./out
   ```
   By default this is a strict production run: real backends must be available. For a development smoke run against the local sample bundle:
   ```bash
   python -m dtm_from_mapillary.cli.pipeline run \
     --dataset-dir data/sample_dataset \
     --imagery-root data/sample_dataset/imagery \
     --out-dir data/sample_dataset/outputs
   ```
   Validate the local sample bundle without running the full pipeline:
   ```bash
   python scripts/check_sample_dataset.py --dataset-dir data/sample_dataset
   ```

   Strict sample QA run against the tracked reference raster:
   ```bash
   COLMAP_DOCKER_IMAGE=colmap/colmap:latest \
   python -m dtm_from_mapillary.cli.pipeline run \
     --dataset-dir data/sample_dataset \
     --imagery-root data/sample_dataset/imagery \
     --reference-dtm qa/data/qa_dtm.tif \
     --out-dir data/sample_dataset/outputs/production_qa \
     --enforce-breaklines
   ```
   
   **Optional: Enable learned uncertainty calibration:**
   ```bash
   python -m dtm_from_mapillary.cli.pipeline run \
     --aoi-bbox "-122.45,37.76,-122.41,37.79" \
     --out-dir ./out \
     --use-learned-uncertainty \
     --uncertainty-model-path ./models/uncertainty.pkl
   ```
   
   **Optional: Enable breakline enforcement for curb preservation:**
   ```bash
   python -m dtm_from_mapillary.cli.pipeline run \
     --aoi-bbox "-122.45,37.76,-122.41,37.79" \
     --out-dir ./out \
     --enforce-breaklines
   ```

   **Optional: Prefetch Mapillary thumbnails (reduces API churn):**
   ```bash
   python -m dtm_from_mapillary.cli.pipeline run \
     --aoi-bbox "-122.45,37.76,-122.41,37.79" \
     --out-dir ./out \
     --cache-imagery \
     --imagery-per-sequence 3
   ```

   **Optional: Tune COLMAP runtime (threads/GPU):**
   ```bash
   python -m dtm_from_mapillary.cli.pipeline run \
     --aoi-bbox "-122.45,37.76,-122.41,37.79" \
     --out-dir ./out \
     --colmap-threads 12 \
     --colmap-use-gpu
   ```
   > Defaults: `--colmap-threads 8`, `--no-colmap-use-gpu`

   **Optional: Use Legacy OpenCV VO instead of DIM:**
   ```bash
   python -m dtm_from_mapillary.cli.pipeline run \
     --aoi-bbox "-122.45,37.76,-122.41,37.79" \
     --out-dir ./out \
     --legacy-vo
   ```
   > By default, the pipeline uses Deep-Image-Matching (SuperPoint+LightGlue) for Track C. Use `--legacy-vo` to fall back to the lightweight CPU-bound OpenCV ORB tracker.

   **Optional: Run monodepth with a real model (TorchScript):**
   ```bash
   MONODEPTH_MODEL_PATH=./models/midas.torchscript.pt \
   python -m dtm_from_mapillary.cli.pipeline run \
     --aoi-bbox "-122.45,37.76,-122.41,37.79" \
     --out-dir ./out
   ```
   > Set `MONODEPTH_DEVICE=cuda` and (optionally) `MONODEPTH_USE_GPU=1` to leverage GPUs when PyTorch is available.

   **Combined: All advanced features:**
   ```bash
   python -m dtm_from_mapillary.cli.pipeline run \
     --aoi-bbox "-122.45,37.76,-122.41,37.79" \
     --out-dir ./out \
     --use-learned-uncertainty \
     --uncertainty-model-path ./models/uncertainty.pkl \
     --enforce-breaklines \
     --cache-imagery \
     --imagery-per-sequence 3
   ```

5. **Outputs** (ellipsoidal heights):
   - `out/dtm_0p5m_ellipsoid.tif`
   - `out/slope_deg.tif`, `out/slope_pct.tif`
   - `out/confidence.tif`
   - `out/ground_points.laz`
   - `out/report.html`
   - `out/manifest.json`
   - `out/qa/qa_summary.json`
   - `out/qa/dz.tif`, `out/qa/abs_dz.tif`, `out/qa/slope_diff_deg.tif` when `--reference-dtm` is supplied
   - Cache root (metadata & imagery): `cache/mapillary/`

> Advanced: To replay a canned OpenSfM reconstruction without invoking the binary, set `OPEN_SFM_FIXTURE=qa/data/opensfm_fixture/reconstruction.json`.

---

## Design Principles

- **Redundancy everywhere**: two SfM stacks (OpenSfM, COLMAP) + VO + mono-depth/plane-sweep; keep only **consensus** (height & slope agreement).
- **Metric scale without external DTMs**: constant camera height per sequence (1–3 m), **GNSS distance** consistency, and **footpoint anchors** from vertical objects.
- **Ground-only focus**: 3D semantic voting from per-image ground masks; reject dynamics (vehicles, pedestrians).
- **Slope fidelity first**: lower-envelope fusion (resist vehicles), plane-fit slope, edge-aware smoothing, curb breaklines, TIN respecting crowns/curbs.
- **OSM corridor**: OSMnx-derived street buffer defines processing mask; **no extrapolation** beyond corridor + `MAX_TIN_EXTRAPOLATION_M` (default 5 m). Always fill **inner blocks** (holes) inside corridor polygons.
- **QA-first**: inter-stack agreement maps, uncertainty, external checkpoints as **hold-out** only.
- **Licensing**: comply with Mapillary terms; store attribution where required.

---

## Key Configuration (see `constants.py`)

- `GRID_RES_M = 0.5`
- `H_MIN_M = 1.0`, `H_MAX_M = 3.0`
- `CORRIDOR_HALF_W_M = 25` (buffer around OSM streets)
- `MAX_TIN_EXTRAPOLATION_M = 5`
- `INCLUDE_INNER_BLOCKS = True`
- `EXCLUDE_ELEVATED_STRUCTURES = True`
- `DZ_MAX_M = 0.25`, `DSLOPE_MAX_DEG = 2.0` for consensus gating

---

## External Data/Tools

- **Mapillary Graph API v4** for images, sequences, detections; **Vector tiles** for discovery.
- **OpenSfM** & **COLMAP** for independent reconstructions.
- **OSMnx** to derive road corridor polygons from OSM.
- Optional coarse DEM only for **initialization/QA** (never to write DTM heights).

---

## Running Pieces Individually

- **Coverage & ingestion**: `api/tiles.py`, `api/mapillary_client.py`, `ingest/sequence_scan.py`
- **Filtering** (car-only): `ingest/sequence_filter.py`
- **Geometry**: `geom/*`
- **Semantics**: `semantics/*`
- **Fusion/Surface**: `fusion/*`, `ground/*`
- **OSM corridor**: `osm/osmnx_utils.py`
- **QA/Reports**: `qa/*`

See **[docs/ROADMAP.md](docs/ROADMAP.md)** for step-by-step implementation tasks and acceptance checks.

---

## Notes on Elevated Structures

Automatic masking uses multi-cue signals:
- **Double-layer parallax** detection (two distinct planes at different heights).
- **Abrupt height discontinuities** unconnected to terrain slope.
- OSM tags for bridges/tunnels where available.
Masked regions are excluded from DTM fusion or flagged low-confidence.

---

## Geoid (Optional Post-Processing)

Use `io/geoutils.py` to apply geoid corrections (EGM96/2008) to convert ellipsoidal heights to orthometric heights if needed.

---

## Attribution & Terms

Respect Mapillary’s terms and attribution requirements. OSM data is © OpenStreetMap contributors. See `ROADMAP.md` for QA steps and validation using official, held-out datasets.
