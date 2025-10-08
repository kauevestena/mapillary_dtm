# AI Agents Guide for DTM from Mapillary

This guide helps AI coding assistants (like GitHub Copilot, Claude, GPT-4, etc.) understand the project structure and provide more effective assistance when working with this codebase.

---

## 🎯 Project Overview

**DTM from Mapillary** is a high-accuracy Digital Terrain Model (DTM) generation pipeline that:
- Processes Mapillary street-level imagery (car sequences only)
- Generates 0.5m resolution ground-only DTMs with ellipsoidal heights
- Produces slope maps optimized for accessibility mapping
- Uses triple-redundancy approach (OpenSfM + COLMAP + VO/mono-depth)
- Focuses on slope fidelity and cross-validation
- Implements OSM-based corridor processing with TIN extrapolation

**Key Innovation:** Metric scale resolution WITHOUT external DTMs, using constant camera height constraints (1-3m), GNSS deltas, and semantic footpoint anchors.

---

## 🏗️ Architecture & Design Principles

### Core Philosophy
1. **Redundancy First**: Three independent geometry tracks (A, B, C) with consensus voting
2. **Ground-Only Focus**: 3D semantic filtering to extract terrain, rejecting vehicles/pedestrians
3. **Slope Fidelity**: Edge-aware smoothing, curb breaklines, lower-envelope fusion
4. **No External DTM Dependency**: Self-contained metric scale resolution
5. **OSM Corridor Bounds**: Street-focused processing with controlled TIN extrapolation (≤5m)

### Height & Coordinate Systems
- **Primary Output**: Ellipsoidal heights (EPSG:4979)
- **Geoid Correction**: Optional post-processing via `io/geoutils.py`
- **Grid Resolution**: 0.5m (configurable via `constants.GRID_RES_M`)

---

## 🧰 Environment & Tooling

- **Virtual environment (`.venv`)**: Always work inside the repository-local virtual environment `.venv`. If it is missing, create it with `python3 -m venv .venv` and install dependencies via `.venv/bin/pip install -r requirements.txt`. All commands (CLI, scripts, tests, linting) should be executed through `.venv/bin/python` to guarantee consistent dependency usage.
- **Python version**: 3.10+ (actively tested on 3.12).
- **Key packages**: `numpy`, `scipy`, `pyproj`, `shapely`, `rasterio`, `geopandas`, `osmnx`, `opencv-python`, `scikit-image`, `laspy[lazrs]`, `pytest`, plus optional `torch` / `torchvision` for mono-depth experiments.

---

## 📁 Module Structure & Responsibilities

### `docs/` - Project Documentation
- **`ROADMAP.md`**: Detailed implementation tasks, milestones, and acceptance criteria
- **`VERIFICATION_REPORT.md`**: Comprehensive verification of milestone completion status

### `api/` - External Data Access
- **`mapillary_client.py`**: Mapillary Graph API v4 wrapper (images, sequences, vector tiles)
- **`tiles.py`**: Web Mercator tile utilities (Z14 coverage discovery)
- **Dependencies**: `requests`, `mercantile`, `mapbox_vector_tile`

### `ingest/` - Data Acquisition & Filtering
- **`sequence_scan.py`**: Discover and cache sequence metadata
- **`sequence_filter.py`**: Car-only filtering (40-120 km/h speed analysis)
- **`camera_models.py`**: Camera model builders (perspective/fisheye/spherical)
- **Output**: JSONL-cached `FrameMeta` per sequence

### `geom/` - Geometric Reconstruction (Triple Track)
- **Track A**: `sfm_opensfm.py` - Full OpenSfM reconstruction
- **Track B**: `sfm_colmap.py` - Independent COLMAP reconstruction  
- **Track C**: `vo_simplified.py` - Up-to-scale visual odometry chain
- **`height_solver.py`**: Per-sequence metric scale + camera height solver (1-3m range)
- **`anchors.py`**: Vertical object footpoint triangulation from Mapillary detections
- **`utils.py`**: Shared geometry utilities

**Key Constraint**: Sequence-constant camera height `h ∈ [1, 3]m` enforced via robust optimization.

### `semantics/` - Scene Understanding
- **`ground_masks.py`**: Semantic segmentation (road/sidewalk/terrain masks)
- **`curb_edge_lane.py`**: Curb/edge line extraction for breakline preservation
- **Output**: Per-image probability maps (`.npz` format)

### `depth/` - Dense Reconstruction Helpers
- **`monodepth.py`**: Monocular depth estimation (auxiliary, NOT for scale)
- **`plane_sweep_ground.py`**: Ground-focused plane-sweep stereo densification
- **Purpose**: Fill weak-parallax regions; scale from Track A/B/C only

### `ml/` - Machine Learning Components
- **`uncertainty_calibration.py`**: Learned uncertainty estimation from consensus validation
- **`integration.py`**: Pipeline integration helpers for ML features
- **Features**: Replaces heuristic uncertainty with data-driven predictions

### `ground/` - Ground Point Processing
- **`ground_extract_3d.py`**: 3D point voting with semantic masks, uncertainty tagging
- **`recon_consensus.py`**: Voxel-based consensus (≥2 of {A,B,C} agreement)
  - Thresholds: `DZ_MAX_M = 0.25`, `DSLOPE_MAX_DEG = 2.0`
- **`corridor_fill_tin.py`**: Delaunay TIN expansion from corridor to AOI
  - Respects `MAX_TIN_EXTRAPOLATION_M = 5`
  - Always fills inner blocks (holes inside corridor polygons)

### `fusion/` - Surface Generation
- **`heightmap_fusion.py`**: Lower-envelope fusion (25th percentile default) + confidence maps
- **`smoothing_regularization.py`**: Edge-aware smoothing tuned for slope preservation

### `osm/` - Corridor Definition
- **`osmnx_utils.py`**: OSMnx-based street corridor polygon generation
- **Buffer**: `CORRIDOR_HALF_W_M = 25` meters
- **Fallback**: Raw sequence trajectories if OSM unavailable

### `qa/` - Quality Assurance & Validation
- **`qa_internal.py`**: Inter-track agreement maps, slope computation, view-count analysis
- **`qa_external.py`**: Hold-out evaluation vs. reference datasets
- **`reports.py`**: HTML report generation with visualizations
- **`data/`**: Ground truth DTMs for validation (geoidal reference)

### `io/` - Input/Output
- **`readers.py`**: GeoTIFF, LAZ, JSONL parsers
- **`writers.py`**: LAZ (with attributes) and GeoTIFF outputs
- **`geoutils.py`**: Geoid correction utilities (EGM96/2008)

### `tiling/` - Spatial Partitioning
- **`tiler.py`**: AOI tiling for large-area processing

### `cli/` - Command-Line Interface
- **`pipeline.py`**: Typer-based orchestrator for full pipeline execution

---

## 🔑 Key Constants (see `constants.py`)

```python
GRID_RES_M = 0.5                    # Output DTM resolution
H_MIN_M = 1.0                       # Min camera height (meters)
H_MAX_M = 3.0                       # Max camera height (meters)
CORRIDOR_HALF_W_M = 25              # OSM street buffer (meters)
MAX_TIN_EXTRAPOLATION_M = 5         # TIN fill limit beyond corridor
INCLUDE_INNER_BLOCKS = True         # Fill holes inside corridor
EXCLUDE_ELEVATED_STRUCTURES = True  # Auto-mask bridges/overpasses
DZ_MAX_M = 0.25                     # Consensus height tolerance
DSLOPE_MAX_DEG = 2.0                # Consensus slope tolerance (degrees)
```

---

## 🔄 Pipeline Workflow (Milestone-Based)

### Phase 1: Discovery & Ingestion
1. Convert AOI bbox to Z14 tiles → discover sequences
2. Filter car-only sequences (speed analysis)
3. Build camera models

### Phase 2: Semantic Processing
4. Generate ground masks (road/sidewalk/terrain)
5. Extract curb/edge lines (optional)

### Phase 3: Geometric Reconstruction (Parallel Tracks)
6. **Track A**: OpenSfM reconstruction
7. **Track B**: COLMAP reconstruction  
8. **Track C**: Visual odometry chain
9. Find anchors (footpoints from vertical objects)
10. Solve per-sequence scale + camera height

### Phase 4: Ground Extraction & Consensus
11. Label 3D points with ground masks → filter
12. Voxelize to 0.5m cells → require ≥2/3 tracks agree
13. Dense mono-depth/plane-sweep for weak-parallax regions

### Phase 5: Corridor & TIN Expansion
14. Build OSM corridor polygon (OSMnx)
15. Create Delaunay TIN from consensus points
16. Sample outside corridor (≤5m extrapolation)
17. Fill inner blocks (holes)

### Phase 6: Fusion & Surface Refinement
18. Lower-envelope fusion (resist vehicle artifacts)
19. Edge-aware smoothing (preserve crowns/curbs)
20. Compute slope maps (degree & percentage)

### Phase 7: QA & Reporting
21. Generate agreement maps (height/slope/view-count)
22. Compare to external reference DTM (hold-out)
23. Create HTML report with metrics & visualizations

---

## 🧪 Testing Strategy

### Unit Tests (`tests/`)
- **`test_camera_models.py`**: Camera projection/unprojection accuracy
- **`test_sequence_filter.py`**: Speed computation from raw positions
- **`test_ground_masks.py`**: Semantic segmentation coverage checks
- **`test_height_solver.py`**: Scale/height optimization with synthetic data
- **`test_geometry_scaffolding.py`**: Pose estimation accuracy
- **`test_curb_edge_lane.py`**: Breakline extraction quality

### Acceptance Criteria (per Milestone)
- **SfM Tracks**: Reprojection RMSE < 1.5 px; ≥90% frame coverage
- **Camera Height**: h ∈ [1.2, 2.5]m typical range
- **Scale Agreement**: A/B scales within 1% difference
- **Ground Density**: > 2 pts/m² in corridor
- **Slope Fidelity**: Curbs/crowns visually intact; realistic continuity
- **TIN Extrapolation**: No fills beyond 5m from corridor edge

---

## 🚨 Common Pitfalls & Solutions

### Problem: Weak Parallax on Ground
**Solution**: Use anchors + plane-sweep + mono-depth densification; enforce constant camera height constraint.

### Problem: Rolling Shutter / Bad Headings
**Solution**: Local bundle adjustment windows; RS-aware models where possible.

### Problem: Dynamic Scenes (vehicles, pedestrians)
**Solution**: Strong semantic filtering via 3D voting; reject non-ground points.

### Problem: Elevated Structures (bridges)
**Solution**: Automatic masking via multi-cue detection (double-layer parallax, height discontinuities) + OSM tags.

### Problem: Missing OSM Coverage
**Solution**: Fallback corridor from raw sequence trajectories.

### Problem: API Rate Limits
**Solution**: Tile-based caching; exponential backoff/retry logic.

---

## 🔍 When Making Changes

### Adding New Features
1. Check `docs/ROADMAP.md` for alignment with project goals
2. Update corresponding test file in `tests/`
3. Add constants to `constants.py` if configurable
4. Update acceptance criteria
5. Document in relevant module docstring

### Modifying Geometric Tracks
- Maintain independence between A/B/C tracks
- Never cross-contaminate scale information
- Update `geom/height_solver.py` if constraints change

### Changing Fusion Logic
- Test on elevated-structure scenarios (bridges)
- Verify slope fidelity on known grades/crowns
- Check curb/edge preservation

### Updating QA Metrics
- Add to `qa/reports.py` HTML generation
- Include in pipeline manifest output
- Document acceptance thresholds

---

## 📚 External Dependencies & Tools

### Required
- **OpenSfM**: Structure-from-Motion (Track A)
- **COLMAP**: Structure-from-Motion (Track B)
- **OSMnx**: OSM road network extraction
- **Mapillary Graph API v4**: Image/sequence metadata

### Optional
- **PyTorch/TorchVision**: Monocular depth estimation
- **EGM96/2008 Geoid Models**: Orthometric height conversion

---

## 🎓 Domain Knowledge Context

### Photogrammetry Concepts
- **Bundle Adjustment**: Simultaneous optimization of camera poses + 3D points
- **Parallax**: Apparent position change between viewpoints (weak for ground in car sequences)
- **Reprojection Error**: Distance between observed and projected image points (quality metric)

### DTM vs DSM
- **DTM (Digital Terrain Model)**: Ground surface only
- **DSM (Digital Surface Model)**: First-return surface (includes buildings/trees)
- **This project**: Pure DTM via semantic filtering

### Slope Representation
- **Degrees**: Angular measure (0° flat, 90° vertical)
- **Percentage**: Rise/run × 100 (10% = 5.7°)
- **Accessibility**: Typically concerns slopes > 5% (2.86°)

### Coordinate Systems
- **Ellipsoidal Height**: Distance above WGS84 ellipsoid
- **Orthometric Height**: Distance above geoid (mean sea level)
- **Geoid-Ellipsoid Separation**: ~30m typical; varies by location

---

## 🤝 Contribution Guidelines

### Code Style
- Follow existing patterns in module
- Use type hints for function signatures
- Add docstrings (Google style) for public functions
- Keep functions focused (single responsibility)

### Commit Messages
Reference milestone/task from `docs/ROADMAP.md` when applicable:
```
M6: Implement plane-sweep ground densifier

- Add depth/plane_sweep_ground.py with cost-volume approach
- Integrate with ground_extract_3d.py consensus voting
- Tests show 2.5x density improvement in low-parallax zones
```

### Pull Requests
- Link to `docs/ROADMAP.md` task
- Include acceptance criteria verification
- Add/update tests
- Update README.md if user-facing changes

---

## 🛠️ Quick Reference for AI Agents

### When User Asks About...

**"How to add a new sequence filter?"**
→ Check `ingest/sequence_filter.py`, follow car-speed filter pattern, update `pipeline.py` orchestration

**"Improve slope accuracy"**
→ Focus on `fusion/smoothing_regularization.py` (edge-aware params) and `semantics/curb_edge_lane.py` (breaklines)

**"Scale resolution issues"**
→ Investigate `geom/height_solver.py` (robust loss, anchor weights) and `geom/anchors.py` (footpoint quality)

**"Add new QA metric"**
→ Implement in `qa/qa_internal.py`, add to manifest in `pipeline.py`, visualize in `qa/reports.py`

**"Handle new camera type"**
→ Extend `ingest/camera_models.py` for new distortion model, ensure OpenSfM/COLMAP compatibility

**"Optimize performance"**
→ Check `tiling/tiler.py` for spatial partitioning; consider GPU acceleration for `depth/` and `semantics/` modules

**"Debug consensus failures"**
→ Visualize agreement maps from `ground/recon_consensus.py`; check `DZ_MAX_M`/`DSLOPE_MAX_DEG` thresholds

**"Where is the roadmap/implementation plan?"**
→ See **[docs/ROADMAP.md](docs/ROADMAP.md)** for detailed tasks and milestones

**"How do I verify implementation status?"**
→ See **[docs/VERIFICATION_REPORT.md](docs/VERIFICATION_REPORT.md)** for complete verification report

---

## 📖 Further Reading

- **[docs/ROADMAP.md](docs/ROADMAP.md)** - Detailed implementation tasks & acceptance criteria
- **[docs/VERIFICATION_REPORT.md](docs/VERIFICATION_REPORT.md)** - Complete verification of milestone completion
- [README.md](README.md) - User-facing documentation & quickstart
- [qa/data/readme.md](qa/data/readme.md) - QA dataset descriptions
- **Mapillary API Docs**: https://www.mapillary.com/developer/api-documentation
- **OSMnx Documentation**: https://osmnx.readthedocs.io/
- **OpenSfM**: https://github.com/mapillary/OpenSfM
- **COLMAP**: https://colmap.github.io/

---

## 🚀 Getting Started (for AI Agents)

When assisting with this project:

1. **Read** `README.md` and `docs/ROADMAP.md` first for context
2. **Check** `constants.py` for configurable parameters
3. **Understand** the triple-track (A/B/C) redundancy philosophy
4. **Respect** the no-external-DTM constraint
5. **Preserve** slope fidelity in any fusion/smoothing changes
6. **Test** changes against acceptance criteria from `docs/ROADMAP.md`
7. **Document** assumptions and trade-offs in code comments
8. **Verify** implementation status in `docs/VERIFICATION_REPORT.md` if needed

---

*Last Updated: October 2025*  
*Project Status: Active Development - See ROADMAP.md for current milestone*
