# Download Script Improvements

**Date:** October 20, 2025

## Summary

Three key improvements have been implemented to the sample data download script to improve debugging and full pipeline testing:

## 1. Rejected Sequences Tracking (filtered_out_sequences.json)

### Purpose
For debugging filter behavior - helps understand why sequences were rejected.

### Location
`data/sample_dataset/sequences/filtered_out_sequences.json`

### Content Structure
```json
{
  "total_sequences_rejected": 28,
  "filter_criteria": {
    "min_speed_kmh": 30.0,
    "max_speed_kmh": 120.0
  },
  "rejection_details": {
    "sequence_id": {
      "frame_count": 150,
      "first_captured": 1562419994413,
      "last_captured": 1562420172551,
      "camera_type": "fisheye",
      "speed_statistics_kmh": {
        "min_kmh": 12.3,
        "q1_kmh": 15.8,
        "median_kmh": 18.2,
        "q3_kmh": 21.5,
        "max_kmh": 25.7,
        "mean_kmh": 18.9,
        "std_kmh": 4.2,
        "sample_count": 149
      },
      "rejection_reason": "max_speed (25.7 km/h) < threshold (30.0 km/h)"
    }
  }
}
```

### Benefits
- **Debug filters:** See exactly why sequences were rejected
- **Validate thresholds:** Check if speed thresholds are too strict/loose
- **Identify edge cases:** Find sequences near threshold boundaries
- **Quality control:** Detect data quality issues in rejected sequences

## 2. Lower Minimum Speed Threshold (30 km/h)

### Previous Value
`MIN_SPEED_KMH = 40.0`

### New Value
`MIN_SPEED_KMH = 30.0`

### Rationale
- **Captures slower urban driving:** Residential streets, traffic congestion
- **Still excludes pedestrian/bicycle:** Typically < 25 km/h
- **Better urban coverage:** More realistic threshold for city street mapping
- **Documented in code:** Clear explanation in `constants.py`

### Impact
- More sequences will pass the filter in dense urban areas
- Better coverage of residential streets
- Maintains clear separation from walking/cycling speeds

### Files Updated
- `constants.py` - Updated constant with documentation
- `scripts/download_sample_data_impl.py` - Updated default argument
- `README.md` template - Updated documentation

## 3. Download ALL Images by Default

### Previous Behavior
- `--images-per-sequence` defaulted to 5
- Required explicit value to download more

### New Behavior
- `--images-per-sequence` defaults to `None` (ALL images)
- Can override with specific limit if needed
- `--cache-imagery` defaults to `True`
- Added `--no-cache-imagery` flag to skip downloads

### Rationale
**Sample dataset purpose:** Test the full pipeline with realistic data density

**Benefits of high density:**
- Better geometric reconstruction (more views per point)
- Stronger consensus voting (more tracks to validate)
- More complete corridor coverage
- Realistic pipeline performance testing
- Better weak-parallax handling

**SfM/photogrammetry benefits:**
- Tighter baselines improve ground reconstruction
- More views = better uncertainty estimation
- Dense coverage = fewer holes in point clouds
- Better scale resolution from GNSS deltas

### Updated CLI Examples

```bash
# Default: Download ALL images from kept sequences (recommended)
python scripts/download_sample_data.py --cache-imagery

# Metadata only (skip imagery)
python scripts/download_sample_data.py --no-cache-imagery

# Limit images per sequence (faster, less dense)
python scripts/download_sample_data.py --cache-imagery --images-per-sequence 10
```

### Files Updated
- `scripts/download_sample_data_impl.py`:
  - Updated `download_imagery_for_sequences()` to handle `None` (all images)
  - Changed default for `--images-per-sequence` to `None`
  - Added `--no-cache-imagery` flag
  - Updated help text and examples
  - Enhanced log messages to show "ALL images" vs. limited

## Implementation Details

### New Function: `save_filtered_out_sequences()`

**Purpose:** Save rejected sequence metadata with speed statistics

**Arguments:**
- `base_dir`: Output directory
- `all_sequences`: All discovered sequences
- `kept_sequences`: Sequences that passed filter
- `speed_stats`: Speed statistics per sequence
- `filter_criteria`: Dict with min/max speed thresholds

**Output:** `sequences/filtered_out_sequences.json`

**Features:**
- Converts `SpeedStatistics` NamedTuple to dict
- Generates human-readable rejection reasons
- Includes full speed statistics for analysis
- Handles missing speed data gracefully

### Modified Function: `download_imagery_for_sequences()`

**New signature:**
```python
def download_imagery_for_sequences(
    gdf, 
    base_dir: Path, 
    per_sequence: int | None  # None = all images
) -> Dict[str, int]:
```

**Changes:**
- `per_sequence` now accepts `None` for unlimited
- Uses conditional: `group if per_sequence is None else group.head(per_sequence)`
- Maintains backward compatibility with explicit limits
- Enhanced docstring

### Modified Function: `save_filtered_sequences()`

**No changes to signature** - already had `speed_stats` parameter from previous update

**Related update:** Now called alongside `save_filtered_out_sequences()` for complete tracking

## Testing Recommendations

1. **Run download script with defaults:**
   ```bash
   MAPILLARY_TOKEN=$(cat mapillary_token) .venv/bin/python scripts/download_sample_data.py
   ```

2. **Verify outputs:**
   - Check `filtered_sequences.json` has speed statistics
   - Check `filtered_out_sequences.json` exists with rejection reasons
   - Verify imagery directory has ALL images from kept sequences

3. **Inspect rejection reasons:**
   ```bash
   cat data/sample_dataset/sequences/filtered_out_sequences.json | jq '.rejection_details[].rejection_reason' | sort | uniq -c
   ```

4. **Compare kept vs. rejected statistics:**
   ```bash
   # Average max speed of kept sequences
   cat data/sample_dataset/sequences/filtered_sequences.json | jq '.sequence_details[].speed_statistics_kmh.max_kmh' | awk '{sum+=$1; n++} END {print sum/n}'
   
   # Average max speed of rejected sequences
   cat data/sample_dataset/sequences/filtered_out_sequences.json | jq '.rejection_details[].speed_statistics_kmh.max_kmh' | awk '{sum+=$1; n++} END {print sum/n}'
   ```

## Backward Compatibility

✅ **Fully backward compatible**

- Old flags still work (`--cache-imagery`, `--images-per-sequence`)
- New flags are optional (`--no-cache-imagery`)
- Defaults changed to recommended values but can be overridden
- No breaking changes to function signatures (added optional parameters)

## Documentation Updates

Updated files:
- `constants.py` - Added comment explaining 30 km/h threshold
- `scripts/download_sample_data_impl.py` - Updated epilog examples
- Generated `README.md` - Mentions both JSON files with explanation
- This document - Complete change summary

## Future Enhancements

Potential improvements for consideration:

1. **Speed histogram visualization:** Generate plots of speed distributions
2. **Interactive filter tuning:** Tool to visualize effect of threshold changes
3. **Multi-criteria filtering:** Beyond speed (e.g., quality_score, time of day)
4. **Automatic threshold suggestion:** Based on distribution analysis
5. **Comparison reports:** Side-by-side stats for kept vs. rejected

---

*Implementation completed: October 20, 2025*  
*Ready for testing with full pipeline*
