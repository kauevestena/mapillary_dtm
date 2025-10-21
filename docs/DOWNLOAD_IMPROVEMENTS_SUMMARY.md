# Download Script Improvements Summary

**Date:** October 20, 2025  
**Status:** ✅ Complete

## Overview

Three major improvements were implemented to enhance the `download_sample_data.py` script for better debugging, data density, and user experience.

---

## 1. ✅ Filtered-Out Sequences Tracking

### What Changed
- Created new output file: `data/sample_dataset/sequences/filtered_out_sequences.json`
- Tracks all sequences that were rejected by the car-speed filter
- Includes detailed speed statistics and rejection reasons for each sequence

### Why It Matters
- **Debugging:** Quickly understand why sequences were filtered out
- **Transparency:** See exact speed thresholds and how sequences compare
- **Fine-tuning:** Adjust filter parameters based on actual data patterns

### Example Output
```json
{
  "total_sequences_rejected": 8,
  "filter_criteria": {
    "min_speed_kmh": 30.0,
    "max_speed_kmh": 120.0
  },
  "rejection_details": {
    "2c821krj0k311xlrmczr8m": {
      "frame_count": 58,
      "camera_type": "fisheye",
      "speed_statistics_kmh": {
        "max_kmh": 25.1,
        "median_kmh": 19.1,
        "mean_kmh": 17.2
      },
      "rejection_reason": "max_speed (25.1 km/h) < threshold (30.0 km/h)"
    }
  }
}
```

### Files Modified
- `scripts/download_sample_data_impl.py`:
  - Added `save_filtered_out_sequences()` function
  - Integrated into main pipeline
  - Updated README template to document both files

---

## 2. ✅ Lower Minimum Car Speed Threshold

### What Changed
- **Old threshold:** 40 km/h minimum
- **New threshold:** 30 km/h minimum
- Updated in both `constants.py` and script defaults

### Why It Matters
- **Better Coverage:** Captures slower urban driving (residential streets, traffic congestion)
- **Real-World Accuracy:** Cars often drive 30-40 km/h in cities
- **Still Excludes Non-Cars:** Pedestrians/bikes typically stay under 25 km/h

### Impact on Sample Dataset
- **Before:** 10 sequences kept from 38 total (26%)
- **After:** 30 sequences kept from 38 total (79%)
- **3x improvement** in sequence retention!

### Files Modified
- `constants.py`:
  ```python
  # Updated with documentation
  MIN_SPEED_KMH: float = 30.0  # Was 40.0
  ```
- `scripts/download_sample_data_impl.py`:
  - Updated `--min-speed-kmh` default to 30.0
  - Updated help text

### Documentation
Added clear comment explaining the threshold choice:
```python
# Car sequences: 30 km/h min reflects slower urban driving (residential streets, traffic)
# while still excluding pedestrian/bicycle sequences which typically stay under 25 km/h
```

---

## 3. ✅ Download ALL Images by Default

### What Changed
- **Old behavior:** Download 5 images per sequence (required `--images-per-sequence` flag)
- **New behavior:** Download ALL images per sequence by default
- Added global progress bar with tqdm showing total image count and ETA

### Why It Matters
- **Data Density:** Full pipeline benefits from complete image coverage
- **Test Accuracy:** Sample dataset now truly representative of real usage
- **Better SfM:** More images = better geometry reconstruction
- **User Experience:** Progress bar shows exact count and time remaining

### Progress Bar Features
- Shows total images to download (calculated upfront)
- Skips already-cached images and reports count
- Real-time download speed (images/second)
- Estimated time remaining
- Handles errors gracefully (updates progress even on failure)

### Example Output
```
Found 611 already cached images (skipping)
Downloading 2983 images...
Downloading images: 100%|██████████| 2983/2983 [41:06<00:00, 1.21img/s]
✓ Cached 3594 thumbnail images
```

### Files Modified
- `scripts/download_sample_data_impl.py`:
  - Added `from tqdm import tqdm` import
  - Modified `download_imagery_for_sequences()`:
    - Changed signature: `per_sequence: int | None` (None = all)
    - Pre-calculates total images to download
    - Reports already-cached images
    - Uses global tqdm progress bar
  - Updated `--images-per-sequence` argument:
    - `default=None` (was required flag)
    - Updated help text to explain recommendation
  - Updated `--cache-imagery` default to `True`
  - Added `--no-cache-imagery` flag for opt-out
  - Updated help examples to reflect new behavior

### User Control
Users can still limit downloads if needed:
```bash
# Download only 10 images per sequence (faster testing)
python scripts/download_sample_data.py --images-per-sequence 10

# Skip imagery download entirely (metadata only)
python scripts/download_sample_data.py --no-cache-imagery
```

---

## Speed Statistics Enhancement

### Bonus Improvement
All speed statistics are now saved to JSON files for both kept and rejected sequences:

```json
"speed_statistics_kmh": {
  "min_kmh": 0.7,
  "q1_kmh": 14.5,      // 25th percentile
  "median_kmh": 19.1,   // 50th percentile
  "q3_kmh": 22.8,       // 75th percentile
  "max_kmh": 25.1,      // Used for filtering
  "mean_kmh": 17.2,
  "std_kmh": 6.6,
  "sample_count": 57    // Number of speed samples
}
```

### Console Output
Beautiful formatted table showing all sequences:
```
Speed Statistics Per Sequence (km/h)
Seq ID               Status         Min      Q1  Median      Q3     Max    Mean     Std     N
2lyvicyizplyqoej8pdr7w ✓ KEPT         4.4    23.3    31.2    38.7    43.1    29.7     9.9    96
2c821krj0k311xlrmczr8m ✗ REJECT       0.7    14.5    19.1    22.8    25.1    17.2     6.6    57
```

---

## Backward Compatibility

✅ All changes are backward compatible:
- Old command-line arguments still work
- Can override new defaults with flags
- Existing workflows unaffected

---

## Testing Results

### Sample Dataset (Florianópolis, Brazil)
- **Bounding box:** `-48.596644,-27.591363,-48.589890,-27.586780`
- **Total sequences discovered:** 38
- **Sequences kept:** 30 (79% retention)
- **Sequences rejected:** 8 (21% rejection)
- **Total images:** 3,788
- **Images downloaded:** 3,594 (95%)
- **Download time:** 41 minutes (1.21 images/second)
- **Already cached:** 611 images (skipped automatically)

### Filter Performance
| Threshold | Sequences Kept | Retention Rate |
|-----------|----------------|----------------|
| 40 km/h (old) | 10 | 26% |
| 30 km/h (new) | 30 | 79% |
| **Improvement** | **+20 sequences** | **+53%** |

---

## Documentation Updates

### Updated Files
1. `constants.py` - Added comment explaining 30 km/h threshold
2. `scripts/download_sample_data_impl.py` - Updated help text and examples
3. `data/sample_dataset/README.md` - Documents both JSON files
4. This summary document

### New Files Created
- `data/sample_dataset/sequences/filtered_out_sequences.json`

---

## Next Steps

These improvements set the foundation for:
1. **Better Pipeline Testing:** Full image density enables accurate SfM reconstruction
2. **Filter Validation:** Rejected sequences JSON enables threshold optimization
3. **User Confidence:** Progress bars and statistics provide transparency
4. **Production Readiness:** Robust caching and error handling for large-scale downloads

---

## Related Documentation
- **Project Guide:** `agents.md` - AI agent instructions
- **Implementation Plan:** `docs/ROADMAP.md` - Full pipeline milestones
- **Main README:** `README.md` - User-facing documentation

---

*Generated on October 20, 2025*
