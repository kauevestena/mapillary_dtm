# Runtime Baseline Audit (Milestone 0)

This note captures the current state of the runtime pipeline and the decisions needed to turn the existing scaffolding into a deployable stack. It satisfies Milestone 0 of `fix-runtime-roadmap.md`.

## External Tooling & Environment Inventory

| Tool / Service | Role in Pipeline | Proposed Support Target | Packaging / Installation Notes | Current Status |
| --- | --- | --- | --- | --- |
| Python runtime | Base environment for CLI & libraries | 3.11–3.12 (project docs currently cite 3.12.3) | Ship `pyenv` + `requirements.txt`; verify C++ build tools for `triangle`, `laspy` extras | In use locally; needs reproducible env spec (Milestone 1) |
| Mapillary Graph API v4 | Real-world imagery & metadata source | No version pin (Graph API); require OAuth token with `images:read` scope | Token sourced from `MAPILLARY_TOKEN` env var or `mapillary_token` file | **Live integration already implemented** (`api/mapillary_client.py`) |
| Mapillary vector tiles | Coverage discovery | Tiles API v2 | Cache under `cache/mapillary/metadata` to reduce requests | **Live integration already implemented** |
| OpenSfM runner | Track A sparse/dense reconstruction | Docker image `mapillary/opensfm:latest` (to be pinned once validated) | Prefer containerized invocation to avoid system-wide deps; needs large tmp volume | **Adapter scaffolded** — `geom/opensfm_adapter.py` loads fixtures, binary path TBD |
| COLMAP CLI | Track B reconstruction parity | Release 3.8 (CUDA build optional) | Expose binary via PATH; confirm CUDA ≥ 11.8 for GPU features | **Adapter available** — fixture loader + CLI knobs (`--colmap-threads`, `--colmap-use-gpu`) |
| CUDA toolkit & GPU drivers | Optional acceleration for COLMAP & dense depth | CUDA 12.1 + driver 535+ (aligns with PyTorch 2.2 LTS) | Required only when enabling GPU paths; document CPU fallback | **Planned** — current code paths default to CPU stubs |
| PyTorch + torchvision | Learned mono-depth & uncertainty calibration | PyTorch 2.2 + torchvision 0.17 (CPU by default) | Optional extras in `requirements.txt`; guard runtime errors when unavailable | Imported in code but all usages are synthetic placeholders |
| Rasterio / GDAL stack | GeoTIFF I/O | rasterio 1.3 + GDAL 3.6 | Wheels cover most platforms; fallback `.npy` writer already implemented | Present with CPU-based fallback |
| laspy + LAZrs | Point cloud export | laspy 2.5 + lazrs 0.6 | Requires Rust toolchain when building from source; `.npz` fallback present | Present with CPU-based fallback |

## Reality Check — Core Pipeline Components

| Domain | Module(s) | Current Behavior | Gaps to Production |
| --- | --- | --- | --- |
| Mapillary ingestion | `api/mapillary_client.py`, `ingest/sequence_scan.py`, `ingest/sequence_filter.py` | Connects to live Graph API, performs bbox discovery, caching, filtering | Need rate-limit guards, retry telemetry, fixture cassette for tests |
| SfM (Track A) | `geom/sfm_opensfm.py`, `geom/opensfm_adapter.py` | Attempts real OpenSfM via adapter, falls back to synthetic scaffolding; fixture support available | Wire full binary invocation + imagery staging, extend tests to cover real outputs |
| SfM (Track B) | `geom/sfm_colmap.py`, `geom/colmap_adapter.py` | Fixture-backed adapter with synthetic fallback; coordinate-frame validation in place | Automate full COLMAP binary invocation & imagery staging |
| Visual odometry | `geom/vo_simplified.py`, `ingest/image_loader.py` | OpenCV ORB + Essential matrix with imagery fallback; synthetic path retained | Extend to full stereo/scale refinement once real imagery available |
| Densification / mono-depth | `depth/monodepth.py`, `depth/plane_sweep_ground.py`, `ground/ground_extract_3d.py` | Procedural depth grids & plane sweep, cached locally | Integrate trained mono-depth model + true plane-sweep; respect GPU availability |
| Breaklines & TIN | `ground/breakline_integration.py`, `ground/corridor_fill_tin.py` | Operates on synthetic point sets; uses `triangle` if installed | Validate on real detections, profile constrained TIN performance |
| QA / Reporting | `qa/*.py`, `io/writers.py` | Generates HTML/npz/np y artifacts with fallbacks | Ensure rasterio/laspy paths exercised; align with ops telemetry plan |

## Target Rollout Run Configuration

- **AOI**: Use the existing Florianópolis, Brazil sample for acceptance testing  
  `aoi_bbox = "-48.596644,-27.591363,-48.589890,-27.586780"` (matches `constants.bbox`).
- **Token management**: Require `MAPILLARY_TOKEN` in env or `mapillary_token` file before launch. Document token scope requirements (`images:read`, vector tiles).
- **Cache layout**: Retain current directories (`cache/mapillary/metadata`, `cache/mapillary/imagery`, `cache/masks`, `cache/depth_mono`) with automated quota pruning (default 2 GB metadata / 8 GB imagery).
- **CLI entry**: `python -m cli.pipeline run --aoi-bbox "$aoi_bbox" --out-dir ./out/fln_baseline --enforce-breaklines` (flags toggled as features land).
- **Expected artifacts** (all routed under `out/fln_baseline/`):
  - GeoTIFFs: `dtm_0p5m_ellipsoid.tif`, `slope_deg.tif`, `confidence.tif`
  - Ground points: `ground_points.laz` (falls back to `.npz` without LAZ support)
  - QA bundle: `qa/agreement_maps.npz`, `report.html`, manifest JSON embedded in report
- **Smoke-data prefetch**: Ship a tiny, pre-downloaded AOI bundle in `qa/data/` for CI to avoid live API calls (follow-up task).

## Immediate Follow-ups

1. Promote environment spec into `docs/` (Milestone 1 seed): lock Python version, system packages, CUDA guidance.
2. Draft interface contracts for OpenSfM / COLMAP adapters (input staging, output parsing) before replacing scaffolds.
3. Capture a mocked Mapillary API cassette to unblock unit tests without network access.
