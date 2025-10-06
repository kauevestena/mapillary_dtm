# ROADMAP — DTM from Mapillary

**Date:** 2025-10-06

This roadmap decomposes the project into **small, testable tasks**, grouped by milestones. Each task maps to a module/function in the scaffold. Acceptance checks are listed to ensure quality, with a focus on **slope fidelity**.

---

## Milestone 0 — Environment & Scaffolding

- [ ] Create Python environment; `pip install -r requirements.txt`.
- [ ] Set `MAPILLARY_TOKEN` env var.
- [ ] Smoke-test imports: `python -m dtm_from_mapillary.cli.pipeline --help`.

**Acceptance:** CLI prints help and exits. No runtime imports fail.

---

## Milestone 1 — Coverage Discovery & Ingestion

**Tasks**
1. `api/tiles.py::bbox_to_z14_tiles` — Convert AOI bbox (lon/lat) to Z14 tiles.
2. `api/mapillary_client.py`:
   - `get_vector_tile(layer,z,x,y)`
   - `list_sequence_ids(bbox)` using vector tiles.
   - `list_image_ids(seq_id)` via Graph API paging.
   - `get_image_meta(image_id)` with fields: `id, sequence, geometry, captured_at, camera_type, camera_parameters, quality_score`.
3. `ingest/sequence_scan.py::discover_sequences` — Assemble `FrameMeta` per sequence.

**Acceptance:** For a small bbox with known coverage, we can list N sequences and ~M images per sequence, saved to JSONL cache. No `computed_*` fields used.

---

## Milestone 2 — Car-Only Filtering & Camera Models

**Tasks**
1. `ingest/sequence_filter.py::filter_car_sequences` — Compute speed from raw positions + timestamps; keep windows with 40–120 km/h.
2. `ingest/camera_models.py::make_opensfm_model` — Build OpenSfM-compatible camera dicts for perspective/fisheye/spherical.

**Acceptance:** Car-only subset produced; camera models serialized. Manual spot-check of sample frames.

---

## Milestone 3 — Semantics (Ground Masks)

**Tasks**
1. `semantics/ground_masks.py` — Load provided masks or run segmentation (road/sidewalk/terrain). Save per-image `.npz` prob-maps.

2. (Optional) `semantics/curb_edge_lane.py` — Extract curb/edge lines for breaklines and slope preservation.

**Acceptance:** Visual overlay for 20 random frames; ground coverage ≥80% on drivable areas; vehicles/people masked out.

---

## Milestone 4 — Geometry Tracks (A/B/C)

**Tasks**
- **A:** `geom/sfm_opensfm.py` — Full OpenSfM reconstruction, no Mapillary seeding.
- **B:** `geom/sfm_colmap.py` — Independent COLMAP reconstruction.
- **C:** `geom/vo_simplified.py` — Up-to-scale VO chain.

**Acceptance:** For a test sequence (≥300 frames), both A and B converge with reprojection RMSE < 1.5 px; VO chain covers ≥90% frames.

---

## Milestone 5 — Anchors & Scale/Height Solver

**Tasks**
1. `geom/anchors.py` — Use Mapillary detections/map-features to find vertical objects, derive **footpoints**, triangulate in A & B poses.
2. `geom/height_solver.py::solve_scale_and_h` — Estimate per-sequence metric scale and constant camera height \(h\in[1,3]\). Use GNSS deltas + anchors; robust loss; return scales & h.

**Acceptance:** On a corridor with clear poles/signs, h in [1.2, 2.5] m; scales consistent between A/B within 1%.

---

## Milestone 6 — Ground-Only 3D Extraction

**Tasks**
1. `ground/ground_extract_3d.py` — Project 3D points into nearby frames; vote with ground masks; label and filter 3D ground points; attach uncertainty & triangulation angle.
2. `depth/monodepth.py` + `depth/plane_sweep_ground.py` — Dense auxiliary ground depth for areas of weak parallax; used only for densification (not scale).

**Acceptance:** Densified ground point cloud with reasonable density (> 2 pts/m² in corridor).

---

## Milestone 7 — Consensus & Fusion

**Tasks**
1. `ground/recon_consensus.py` — Voxelize to 0.5 m cells; require ≥2 of {A,B,C} to agree within `DZ_MAX_M` and slope within `DSLOPE_MAX_DEG`.
2. `fusion/heightmap_fusion.py` — Lower-envelope fusion (25th percentile by default), produce DTM + confidence.
3. `fusion/smoothing_regularization.py` — Edge-aware smoothing tuned for slope fidelity.

**Acceptance:** Visual inspection shows crowns/curbs intact; noise visibly reduced; slope continuity realistic.

---

## Milestone 8 — Corridor→AOI via Delaunay TIN

**Tasks**
1. `osm/osmnx_utils.py` — Build corridor polygon(s) from OSM highways using OSMnx; buffer by `CORRIDOR_HALF_W_M`.

   - Always include **inner blocks** (holes) inside corridor polygons.

2. `ground/corridor_fill_tin.py` — Build TIN from accepted corridor points; sample to AOI grid outside corridor; **limit extrapolation** to `MAX_TIN_EXTRAPOLATION_M`.

3. Elevated-structure masking (automatic) — Integrate with consensus/fusion masks.

**Acceptance:** Outside-corridor cells are filled only where within 5 m outward from corridor footprint; inner blocks filled; elevated decks masked.

---

## Milestone 9 — Slope & QA

**Tasks**
1. `qa/qa_internal.py::slope_from_plane_fit` — Plane-fit slope (deg and %).
2. `qa/qa_internal.py::agreement_maps` — Height/slope disagreement rasters; view-count.
3. `qa/qa_external.py::compare_to_geotiff` — Hold-out evaluation vs. official datasets.

**Acceptance:** RMSE_z and slope RMSE reported; problem tiles flagged; HTML report generated.

---

## Milestone 10 — Packaging & Reproducibility

**Tasks**
1. `io/writers.py` — Write LAZ (with attributes) and GeoTIFFs.

2. `qa/reports.py` — HTML report with summaries, maps, histograms.

3. Manifests: capture constants snapshot, git SHA, token hash.

**Acceptance:** All outputs versioned; re-running with same inputs yields same results within numerical noise.

---

## Stretch Goals

- Breakline enforcement (curbs/medians) in TIN.
- Self-calibration refinement for fisheye/spherical cameras.
- Learned uncertainty calibration for mono-depth.

- GPU acceleration for plane-sweep and segmentation.

---

## Risk Register & Mitigations

- **Weak parallax on ground** → Use anchors + plane-sweep + mono for densification; enforce constant-h.

- **Bad headings / rolling shutter** → Local BA windows, RS-aware models where possible.

- **Dynamic scenes** → Strong semantic filtering.

- **Bridges** → Automatic masking + OSM bridge tags.

- **OSM gaps** → Fallback corridor from raw sequence trajectories.

- **API rate limits** → Tile-based caching; backoff/retry logic.

---

## Acceptance of Final System

- Visual: No obvious vehicle/structure artifacts in DTM; slopes reasonable along grades and crowns.

- Quantitative: Meets announced RMSE targets on held-out checkpoints; slope RMSE within thresholds.

- Reproducible: Deterministic with fixed seeds and inputs.

- Compliant: Mapillary & OSM license terms respected.
