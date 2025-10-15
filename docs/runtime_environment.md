# Runtime Environment Specification

This guide defines the baseline environment required to run the Mapillary DTM pipeline with real data. It extends the Milestone 0 audit with concrete version pins and installation guidance. Treat this as the single source of truth when standing up new machines, CI runners, or container images.

## Python & Toolchain

| Component | Requirement | Notes |
| --- | --- | --- |
| Python | 3.12.3 (recommended), 3.11.8 (minimum) | Create a virtualenv via `pyenv` or `python -m venv`; ensure `pip>=23.2`. |
| Compiler toolchain | GCC ≥ 11, Clang ≥ 14 | Required for packages with native extensions (`triangle`, `laspy[lazrs]`, `rasterio`). |
| Build essentials | `build-essential`, `cmake`, `ninja-build` | Needed to compile COLMAP from source if prebuilt binaries unavailable. |

> Tip: On Ubuntu 22.04+
> ```bash
> sudo apt-get update && sudo apt-get install -y \
>   build-essential cmake ninja-build pkg-config \
>   python3-dev python3-venv python3-pip
> ```

## System Libraries

| Package | Purpose | Ubuntu 22.04 Install |
| --- | --- | --- |
| GDAL + PROJ | `rasterio`, `geopandas`, `osmnx` | `sudo apt-get install -y gdal-bin libgdal-dev libproj-dev` |
| GEOS | Geometry ops for Shapely | Included with GDAL packages |
| Spatialindex | Required by `rtree` (GeoPandas dependency) | `sudo apt-get install -y libspatialindex-dev` |
| libboost-all-dev | COLMAP dependency | `sudo apt-get install -y libboost-all-dev` |
| FreeImage, GFlags, GLog, Eigen, Ceres | COLMAP dependency bundle | `sudo apt-get install -y libfreeimage-dev libgoogle-glog-dev libgflags-dev libeigen3-dev libcgal-dev libceres-dev` |
| OpenCV runtime (`libopencv-core405`) | Optional, accelerates VO experiments | `sudo apt-get install -y libopencv-dev` |
| Rust toolchain (`rustup`) | `lazrs` backend for `laspy` | `curl https://sh.rustup.rs -sSf | sh` |
| CUDA Toolkit 12.1 + Driver ≥ 535 | Optional: GPU acceleration for COLMAP, PyTorch-based modules | See NVIDIA docs; ensure `nvcc --version` reports ≥ 12.1 |

> If GPU hardware is unavailable, keep CUDA packages out of scope; the pipeline automatically falls back to CPU-only paths.

## External Binaries & Containers

| Tool | Version | Install Guidance | Usage |
| --- | --- | --- | --- |
| COLMAP | 3.8 | Prefer official binary release; otherwise build from source with CUDA off/on. Ensure `colmap --help` works. | Track B reconstruction. |
| OpenSfM | Latest Docker image (`mapillary/opensfm:latest`) or pinned commit `2024-05-15` | Pull docker image or install from source with `pip install opensfm @ git+https://github.com/mapillary/OpenSfM@<commit>`. | Track A reconstruction. |
| Docker Engine | 24.0+ (if using OpenSfM container) | Install from Docker docs; add user to `docker` group. | Running OpenSfM container workflows. |
| GNU Parallel (optional) | 20231122 | `sudo apt-get install -y parallel` | Batch invocation helper for SfM jobs. |

## Python Dependency Layers

Use the provided requirement files:

| File | Purpose |
| --- | --- |
| `requirements.txt` | Core runtime stack (ingestion, geometry, QA, CLI). |
| `requirements-optional.txt` | Heavy or optional extras (PyTorch, visualization, notebook tooling). |

Install with:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Optional extras:
pip install -r requirements-optional.txt
```

For GPU-enabled PyTorch replace the wheel specification in `requirements-optional.txt` with the CUDA-specific index (see https://pytorch.org/get-started/locally/).

## Environment Variables & Secrets

| Variable | Description | Required | Notes |
| --- | --- | --- | --- |
| `MAPILLARY_TOKEN` | OAuth token with `images:read` scope | Yes (runtime) | Alternative: define in `.env`, set `MAPILLARY_TOKEN_FILE`, or place token in repo root `mapillary_token`. |
| `OPEN_SFM_BIN` | Path to `opensfm_run_all` (or wrapper) | No | Overrides the binary used by the OpenSfM adapter. |
| `OPEN_SFM_FIXTURE` | Path to canned OpenSfM reconstruction | No | Enables fixture-driven runs without invoking the binary. |
| `OPEN_SFM_FORCE_SYNTHETIC` | Force synthetic SfM scaffold | No | Set to `1` to skip adapter attempts entirely. |
| `CUDA_VISIBLE_DEVICES` | GPU selection | No | Set when running GPU-accelerated stages. |
| `DTM_CACHE_ROOT` | Override cache directory root | No | Defaults to `./cache`. |

## Recommended Hardware Baselines

- **CPU**: 8 cores (16 threads) minimum; 32 GB RAM for comfortable COLMAP/OpenSfM runs.
- **GPU** (optional): NVIDIA RTX 3080 (10 GB) or better when enabling dense reconstruction or learned models.
- **Storage**: ≥ 200 GB free SSD for raw imagery, cache, and intermediate artifacts.

## Validation Checklist

Run the following after provisioning a machine:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-optional.txt  # if needed
python scripts/check_env.py --full
```

Ensure that:
1. `colmap --help` succeeds.
2. `docker run --rm mapillary/opensfm:latest opensfm_run_all --help` returns usage info.
3. `nvidia-smi` reports GPUs when CUDA paths are enabled.
4. `pytest` passes when run against smoke fixtures (planned in later milestones).

## Change Management

- Update this document whenever dependency versions change.
- Regenerate lockfiles (if introduced later) in tandem with doc updates.
- Keep `scripts/check_env.py` aligned with the requirements above.
