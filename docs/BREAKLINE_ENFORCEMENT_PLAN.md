# Breakline Enforcement Implementation Plan

**Stretch Goal:** Breakline enforcement (curbs/medians) in TIN  
**Status:** 🏗️ In Progress  
**Date:** 2025-10-08

---

## Overview

Breaklines are linear features (curbs, lane edges, medians) where terrain slope changes sharply. Without enforcement, standard Delaunay TIN interpolation smooths across these features, degrading slope fidelity. This implementation adds **constrained Delaunay triangulation** to preserve breaklines as triangle edges.

### Key Benefits
1. **Slope Fidelity**: Preserves sharp slope changes at curbs/crowns
2. **Accessibility Mapping**: Accurate curb heights for wheelchair routing
3. **Visual Quality**: Natural-looking road profiles with crisp edges
4. **Metric Accuracy**: Prevents height averaging across discontinuities

---

## Architecture

### Current State
- ✅ `semantics/curb_edge_lane.py`: Extracts curb polylines from ground masks
- ✅ `ground/corridor_fill_tin.py`: Builds Delaunay TIN from consensus points
- ❌ **Missing**: Connection between curb extraction and TIN construction

### Implementation Components

```
semantics/curb_edge_lane.py
    ↓ (CurbLine objects)
ground/breakline_integration.py (NEW)
    ├── project_curbs_to_3d()      # Project 2D image curves to 3D
    ├── merge_breakline_segments() # Build continuous polylines
    └── densify_breaklines()       # Add vertices for TIN
    ↓ (3D breakline vertices + segments)
ground/corridor_fill_tin.py (MODIFIED)
    └── build_constrained_tin()    # Use scipy.spatial.Delaunay with constraints
```

---

## Implementation Tasks

### Task 1: 3D Breakline Projection
**File:** `ground/breakline_integration.py` (new)

**Subtasks:**
1. Load curb polylines from `curb_edge_lane.py`
2. For each curb point (x_img, y_img):
   - Ray-cast from camera through image point
   - Intersect with ground plane (height from nearby consensus points)
   - Store 3D position (X, Y, Z)
3. Filter outliers (use median elevation in local window)

**Input:**
- `CurbLine` objects (normalized image coordinates)
- Camera poses (from Track A/B)
- Consensus ground points (for height reference)

**Output:**
- `Breakline3D` dataclass:
  ```python
  @dataclass
  class Breakline3D:
      seq_id: str
      points: List[Tuple[float, float, float]]  # (X, Y, Z) in ENU
      type: str  # "curb" | "lane_edge" | "median"
      confidence: float
  ```

**Acceptance:**
- Visual overlay: 3D breaklines align with curbs in point cloud
- Heights within ±0.2m of nearby consensus points

---

### Task 2: Breakline Merging & Simplification
**File:** `ground/breakline_integration.py`

**Subtasks:**
1. Merge overlapping segments from multiple images:
   - Use proximity threshold (0.5m horizontal)
   - Average positions with confidence weighting
2. Simplify polylines (Douglas-Peucker with 0.1m tolerance)
3. Enforce minimum segment length (2m) to avoid noise

**Input:**
- Raw `Breakline3D` objects from all sequences

**Output:**
- Merged `Breakline3D` objects (one per physical feature)

**Acceptance:**
- No duplicate curbs within 0.5m
- Polylines smooth but preserve sharp corners
- Total vertex count reduced by 50-70%

---

### Task 3: Breakline Densification
**File:** `ground/breakline_integration.py`

**Subtasks:**
1. Resample breaklines to uniform spacing (≤0.5m between vertices)
2. Ensure vertices exist at breakline intersections
3. Add "constraint edge" metadata for TIN

**Input:**
- Merged `Breakline3D` objects

**Output:**
- Densified vertices + edge connectivity list:
  ```python
  vertices: np.ndarray  # (N, 3) XYZ positions
  edges: List[Tuple[int, int]]  # Vertex index pairs
  ```

**Acceptance:**
- Vertex spacing ≤ 0.5m (matches grid resolution)
- Edge list forms continuous polylines
- No self-intersections

---

### Task 4: Constrained Delaunay TIN
**File:** `ground/corridor_fill_tin.py` (modified)

**Subtasks:**
1. Add `build_constrained_tin()` function:
   - Combine consensus points + breakline vertices
   - Pass edge constraints to `scipy.spatial.Delaunay` or use `triangle` library
   - Verify constraints are preserved as triangle edges
2. Update `TINModel` dataclass to include constraints
3. Modify `sample_outside_corridor()` to respect constraints

**Input:**
- Consensus ground points
- Breakline vertices + edges

**Output:**
- `TINModel` with constrained edges

**Technical Note:**
`scipy.spatial.Delaunay` doesn't natively support edge constraints. Options:
1. **Use `triangle` library** (Shewchuk's Triangle with PSLG support)
2. **Post-process Delaunay**: Insert breakline edges, flip non-Delaunay triangles
3. **Hybrid approach**: Standard Delaunay + edge-aware interpolation

**Recommended:** Use `triangle` library (add to requirements.txt)

**Acceptance:**
- All breakline edges appear as triangle edges (no crossings)
- Visual inspection: TIN respects curbs (no smoothing across)
- Interpolated heights change sharply at breakline edges

---

### Task 5: Pipeline Integration
**File:** `cli/pipeline.py`

**Subtasks:**
1. Add `--enforce-breaklines` CLI flag (default: False)
2. Call breakline projection after consensus
3. Pass breaklines to TIN construction
4. Update manifest to record breakline stats

**CLI Example:**
```bash
python -m dtm_from_mapillary.cli.pipeline run \
  --aoi-bbox "lon_min,lat_min,lon_max,lat_max" \
  --out-dir ./out \
  --enforce-breaklines
```

**Acceptance:**
- Flag toggles breakline enforcement
- Pipeline runs successfully with/without flag
- Manifest includes breakline count and coverage

---

### Task 6: Testing & Validation
**File:** `tests/test_breakline_enforcement.py` (new)

**Test Cases:**
1. `test_project_curbs_to_3d`: Synthetic camera + curb → 3D projection
2. `test_merge_overlapping_segments`: Duplicate removal
3. `test_simplify_polyline`: Douglas-Peucker correctness
4. `test_densify_breaklines`: Uniform resampling
5. `test_constrained_tin`: Triangle library integration
6. `test_breakline_preservation`: Interpolation respects edges
7. `test_curb_height_accuracy`: Curb heights within tolerance
8. `test_pipeline_with_breaklines`: End-to-end run

**Acceptance:**
- All tests pass
- Coverage ≥90% for new code

---

## Data Flow

```
1. Image Acquisition
   ↓
2. Semantic Segmentation → Ground Masks
   ↓
3. Curb Extraction (curb_edge_lane.py)
   ↓ CurbLine (2D image space)
   ↓
4. SfM Reconstruction (Track A/B) → Camera Poses
   ↓
5. Consensus Voting → Ground Points
   ↓
6. **NEW: Breakline Projection (breakline_integration.py)**
   ├── Ray-cast from camera through curb pixels
   ├── Intersect with local ground plane
   └── Filter outliers
   ↓ Breakline3D (3D world space)
   ↓
7. **NEW: Breakline Processing**
   ├── Merge overlapping segments
   ├── Simplify polylines
   └── Densify to grid resolution
   ↓ Vertices + Edge Constraints
   ↓
8. **MODIFIED: Constrained TIN Construction**
   ├── Combine ground points + breakline vertices
   └── Build constrained Delaunay (triangle library)
   ↓ Constrained TINModel
   ↓
9. Grid Sampling (respects constraints)
   ↓
10. Heightmap Fusion
   ↓
11. DTM Output
```

---

## Technical Challenges & Solutions

### Challenge 1: Edge Constraint Library
**Problem:** `scipy.spatial.Delaunay` doesn't support edge constraints

**Solution:** Use `triangle` library (wrapper for Shewchuk's Triangle)
```python
import triangle

# Build PSLG (Planar Straight Line Graph)
pslg = {
    'vertices': np.array([[x1,y1], [x2,y2], ...]),
    'segments': np.array([[0,1], [1,2], ...]),  # Edge constraints
}
tri = triangle.triangulate(pslg, 'pq30a')  # Quality mesh, max area
```

**Dependencies:** Add `triangle` to requirements.txt

---

### Challenge 2: Height Ambiguity at Curbs
**Problem:** Curb has two heights (top and bottom) - which to use?

**Solution:** Use **bottom** (ground level) for terrain model
- Curb top is a building/structure feature (DSM, not DTM)
- Ground mask gradient captures transition zone
- Use prob_band=(0.45, 0.6) to target bottom edge

---

### Challenge 3: Breakline Density vs. TIN Complexity
**Problem:** Too many breakline vertices → TIN explosion

**Solution:** Adaptive densification
- Use `GRID_RES_M` (0.5m) as max spacing
- Simplify with Douglas-Peucker (0.1m tolerance)
- Limit total breakline vertices to 10% of point cloud

---

### Challenge 4: Multi-View Inconsistency
**Problem:** Same curb appears different in different images

**Solution:** Consensus-based merging
- Average positions within 0.5m horizontal window
- Weight by confidence scores
- Reject outliers beyond 2σ from median height

---

### Challenge 5: Breaklines Outside Corridor
**Problem:** TIN extrapolation may extend breaklines unnaturally

**Solution:** Clip breaklines to corridor buffer
- Only enforce constraints within `CORRIDOR_HALF_W_M + MAX_TIN_EXTRAPOLATION_M`
- Beyond corridor, revert to standard Delaunay

---

## Performance Impact

### Computational Cost
- **Curb Projection**: +5-10% runtime (ray-casting)
- **Breakline Merging**: +2-3% runtime (spatial indexing)
- **Constrained TIN**: +10-20% runtime (vs. standard Delaunay)
- **Total**: ~20% increase in total pipeline time

### Memory Overhead
- Breakline vertices: ~5-10% of consensus point count
- Edge constraints: ~2x vertex count (typically)
- **Total**: ~10-15% memory increase

### Quality Improvement
- Curb height accuracy: 0.15m → 0.08m RMSE (estimated)
- Slope continuity: Visual improvement (qualitative)
- Accessibility metrics: Wheelchair routing viable

---

## Configuration Parameters

Add to `constants.py`:

```python
# Breakline enforcement
BREAKLINE_ENABLED = False  # Toggle via CLI
BREAKLINE_PROJ_PROB_BAND = (0.45, 0.6)  # Ground mask gradient range
BREAKLINE_MERGE_DIST_M = 0.5  # Merge segments within this distance
BREAKLINE_SIMPLIFY_TOL_M = 0.1  # Douglas-Peucker tolerance
BREAKLINE_DENSIFY_MAX_SPACING_M = 0.5  # Vertex resampling interval
BREAKLINE_MIN_LENGTH_M = 2.0  # Discard short segments
BREAKLINE_MAX_HEIGHT_DEV_M = 0.3  # Outlier filter threshold
```

---

## Acceptance Criteria

### Visual Inspection
- ✅ Curbs/edges visible as sharp breaks in DTM
- ✅ No smoothing across breaklines
- ✅ Road crown profiles realistic (peaked, not rounded)

### Quantitative Metrics
- ✅ Curb height error < 0.10m RMSE (vs. ground truth)
- ✅ Breakline coverage ≥ 60% of detected curbs
- ✅ TIN edge count includes 100% of breakline segments

### Integration Tests
- ✅ Pipeline runs with `--enforce-breaklines` flag
- ✅ DTM output includes breakline metadata layer
- ✅ QA report shows breakline statistics

### Performance
- ✅ Runtime increase < 25% vs. baseline
- ✅ Memory overhead < 20% vs. baseline

---

## Rollout Plan

### Phase 1: Projection & Merging (Week 1)
1. Implement `breakline_integration.py` (Tasks 1-3)
2. Unit tests for projection and merging
3. Validation with synthetic data

### Phase 2: Constrained TIN (Week 2)
4. Add `triangle` library dependency
5. Implement `build_constrained_tin()` (Task 4)
6. Integration tests with real corridor data

### Phase 3: Pipeline Integration (Week 3)
7. CLI flag and orchestration (Task 5)
8. End-to-end testing (Task 6)
9. Documentation and examples

### Phase 4: Validation & Tuning (Week 4)
10. Real-world testing on diverse sites
11. Parameter tuning (prob_band, tolerances)
12. Performance profiling and optimization

---

## Example Workflow

### 1. Extract Curbs (Already Implemented)
```python
from semantics.curb_edge_lane import extract_curbs_and_lanes

curbs = extract_curbs_and_lanes(
    seqs=sequences,
    mask_dir="cache/masks",
    prob_band=(0.45, 0.6)
)
```

### 2. Project to 3D (NEW)
```python
from ground.breakline_integration import project_curbs_to_3d

breaklines_3d = project_curbs_to_3d(
    curbs=curbs,
    camera_poses=poses_A,
    consensus_points=consensus_results
)
```

### 3. Merge & Densify (NEW)
```python
from ground.breakline_integration import (
    merge_breakline_segments,
    densify_breaklines
)

merged = merge_breakline_segments(breaklines_3d)
vertices, edges = densify_breaklines(merged)
```

### 4. Build Constrained TIN (MODIFIED)
```python
from ground.corridor_fill_tin import build_constrained_tin

tin = build_constrained_tin(
    points=consensus_results,
    breakline_vertices=vertices,
    breakline_edges=edges
)
```

### 5. Sample & Fuse
```python
samples = sample_outside_corridor(
    consensus_points=consensus_results,
    corridor_info=corridor,
    tin=tin  # Uses constrained TIN
)

dtm = fuse_heightmap(consensus_results + samples)
```

---

## Dependencies

### New Requirements
Add to `requirements.txt`:
```
triangle>=20230923  # Constrained Delaunay triangulation
```

### Optional Enhancements
```
rtree>=1.0.0  # Spatial indexing for fast merging
networkx>=3.0  # Breakline graph analysis
```

---

## Documentation Updates

### Files to Update
1. **README.md**: Add breakline enforcement to features list
2. **docs/ROADMAP.md**: Mark stretch goal as in-progress/complete
3. **docs/VERIFICATION_REPORT.md**: Add implementation notes
4. **agents.md**: Document new module structure
5. **NEW: docs/BREAKLINE_ENFORCEMENT_GUIDE.md**: User-facing documentation

---

## Future Enhancements

### Multi-Class Breaklines
- Separate handling for curbs, lane edges, medians
- Type-specific constraints (e.g., median height = curb avg)

### Adaptive Constraint Strength
- Strong constraints for high-confidence curbs
- Soft constraints for uncertain edges
- Learned weights from validation data

### Breakline Quality Metrics
- Coverage map (% of detected curbs enforced)
- Preservation ratio (enforced edges / detected curbs)
- Height agreement (TIN heights vs. curb observations)

### Integration with Learned Uncertainty
- Increase uncertainty near breaklines (transition zones)
- Use breakline presence as feature in ML calibration

---

**Next Step:** Begin implementation with Task 1 (3D Breakline Projection)

