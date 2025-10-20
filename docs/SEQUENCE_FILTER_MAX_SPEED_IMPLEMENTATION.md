# Sequence Filter: Max-Speed Implementation

**Date**: October 20, 2025  
**Status**: ✅ Complete

## Summary

Updated the car sequence filtering logic to match the **original design intent**: judge sequences by their **maximum speed achieved** rather than filtering frame-by-frame.

---

## Changes Made

### 1. `ingest/sequence_filter.py`

**Before (Frame-by-Frame Filtering)**:
- Computed speeds at each frame position
- Filtered out individual frames if their local speed was outside [40, 120] km/h
- Result: Fragmented sequences with only "fast enough" segments

**After (Max-Speed Sequence Filtering)**:
- Computes **all frame-to-frame speeds** in the sequence
- Takes the **maximum speed** across the entire sequence
- **Judges the whole sequence**: Keep if `max_speed ∈ [min_speed_kmh, max_speed_kmh]`
- Then filters individual frames by quality/camera type only

**Key Benefits**:
- **Correctly identifies car sequences**: Cars can slow/stop at intersections but still reach car speeds
- **Rejects pedestrian/bike sequences**: They never reach 40+ km/h
- **Simple & intuitive**: One threshold check per sequence
- **Configurable**: Accepts custom `min_speed_kmh` and `max_speed_kmh` parameters

**Function Signature**:
```python
def filter_car_sequences(
    seqs: Dict[str, List[FrameMeta]],
    min_speed_kmh: float | None = None,  # Default: constants.MIN_SPEED_KMH (40)
    max_speed_kmh: float | None = None,  # Default: constants.MAX_SPEED_KMH (120)
) -> Dict[str, List[FrameMeta]]:
```

### 2. `scripts/download_sample_data_impl.py`

**Added CLI Arguments**:
```bash
--min-speed-kmh 40.0    # Minimum max-speed threshold (default: 40 km/h)
--max-speed-kmh 120.0   # Maximum max-speed threshold (default: 120 km/h)
```

**Example Usage**:
```bash
# Default (40-120 km/h)
python scripts/download_sample_data.py

# Custom thresholds for urban areas (lower speeds accepted)
python scripts/download_sample_data.py --min-speed-kmh 30 --max-speed-kmh 100

# Strict filtering for highways only
python scripts/download_sample_data.py --min-speed-kmh 60 --max-speed-kmh 130
```

**Improved Logging**:
- Shows actual thresholds used: `"Filtering car-only sequences (30-100 km/h max-speed analysis)..."`
- Suggests threshold adjustment if no sequences found
- Debug logs show per-sequence max speeds and decisions

### 3. `tests/test_sequence_filter.py`

**Comprehensive Test Coverage**:
- ✅ `test_filter_car_sequences_keeps_reasonable_speeds` - Car speeds (40-120 km/h) pass
- ✅ `test_filter_car_sequences_drops_too_slow` - Pedestrian/bike speeds (< 40 km/h) rejected
- ✅ `test_filter_car_sequences_drops_too_fast` - Highway speeds (> 120 km/h) rejected
- ✅ `test_filter_car_sequences_drops_low_quality_frames` - Per-frame quality filtering
- ✅ `test_filter_car_sequences_drops_wrong_camera_type` - Camera type filtering
- ✅ `test_filter_car_sequences_custom_thresholds` - Custom min/max thresholds work
- ✅ `test_filter_car_sequences_mixed_speeds` - **Key test**: Sequence with mixed speeds (stops + acceleration) correctly judged by max speed

---

## Logic Flow

### Previous (Incorrect) Approach:
```
For each sequence:
  For each frame in sequence:
    Compute average speed at this frame position
    if speed < MIN or speed > MAX:
      Drop this frame
  Keep sequence if any frames remain
```

**Problem**: A car sequence that stops at a traffic light would have most frames dropped, even though it's clearly a car-mounted camera.

### New (Correct) Approach:
```
For each sequence:
  Compute all frame-to-frame speeds
  max_speed = max(all_speeds)
  
  if max_speed < MIN_SPEED or max_speed > MAX_SPEED:
    Reject entire sequence
    continue
  
  # Sequence passes - now filter by quality/camera
  For each frame in sequence:
    if quality < threshold or camera_type not allowed:
      Drop this frame
  
  Keep sequence if any frames remain
```

**Result**: Car sequences are correctly identified even when stopped/slowing, because they **achieve** car-level speeds at some point.

---

## Speed Threshold Rationale

### Default: 40-120 km/h

| Transport Mode | Typical Speed | Filtered? |
|----------------|--------------|-----------|
| Pedestrian | 4-6 km/h | ❌ Rejected (< 40) |
| Bicycle | 15-25 km/h | ❌ Rejected (< 40) |
| **Urban Car** | **30-60 km/h** | ✅ **Kept** (reaches 40+) |
| **Highway Car** | **80-110 km/h** | ✅ **Kept** |
| High-speed highway | 130-150 km/h | ❌ Rejected (> 120) |

### Why Max Speed Matters for DTM Quality

1. **Camera Height Constraint**: Cars (1-3m) vs bikes/pedestrians (0.5-1.8m)
2. **Stable Mounting**: Vehicle mount = consistent orientation
3. **Parallax Quality**: Car speed → adequate baseline → good ground triangulation
4. **Scene Stability**: Cars avoid high-motion artifacts

---

## Backward Compatibility

✅ **Fully backward compatible**:
- Default parameters use existing `constants.MIN_SPEED_KMH` / `constants.MAX_SPEED_KMH` (40/120)
- Existing code calling `filter_car_sequences(seqs)` works unchanged
- CLI scripts default to same thresholds as before

---

## Testing Results

```bash
$ .venv/bin/python -m pytest tests/test_sequence_filter.py -v

tests/test_sequence_filter.py::test_filter_car_sequences_keeps_reasonable_speeds PASSED      [ 14%]
tests/test_sequence_filter.py::test_filter_car_sequences_drops_too_slow PASSED               [ 28%]
tests/test_sequence_filter.py::test_filter_car_sequences_drops_too_fast PASSED               [ 42%]
tests/test_sequence_filter.py::test_filter_car_sequences_drops_low_quality_frames PASSED     [ 57%]
tests/test_sequence_filter.py::test_filter_car_sequences_drops_wrong_camera_type PASSED      [ 71%]
tests/test_sequence_filter.py::test_filter_car_sequences_custom_thresholds PASSED            [ 85%]
tests/test_sequence_filter.py::test_filter_car_sequences_mixed_speeds PASSED                 [100%]

=============================================================================== 7 passed in 0.61s ===============================================================================
```

---

## Documentation Updates

Updated references in:
- This document (new)
- Inline docstrings in `sequence_filter.py`
- CLI help text in `download_sample_data_impl.py`
- Test docstrings explaining expected behavior

---

## Future Considerations

### Potential Enhancements:
1. **Percentile-based filtering**: Use 90th percentile instead of max to be more robust to GPS noise
2. **Minimum sustained speed**: Require X consecutive frames above threshold
3. **Speed histogram analysis**: Detect bimodal distributions (car + pedestrian in same sequence)
4. **Acceleration analysis**: Additional validation via acceleration profiles

### Current Implementation is Sufficient Because:
- GPS positions are already smoothed by Mapillary
- Outliers filtered by `dt_s <= 0.1` check (ignores stalls/duplicates)
- Quality score already handles noisy data
- Simple max-speed is intuitive and documented

---

## Related Files

- `ingest/sequence_filter.py` - Core filtering logic
- `scripts/download_sample_data_impl.py` - CLI integration
- `tests/test_sequence_filter.py` - Test coverage
- `constants.py` - Default thresholds (`MIN_SPEED_KMH`, `MAX_SPEED_KMH`)

---

**Implementation Status**: ✅ Complete and tested  
**Acceptance Criteria**: ✅ All tests pass, backward compatible, configurable via CLI
