# Metric Scale & Consensus Validation - Failure Modes

This document describes expected failure modes in the metric scale and consensus validation stages of the pipeline, and how the system responds to them.

## Scale Estimation Failures

### Insufficient GPS Coverage

**Symptoms:**
- Very small or zero GPS step sizes (< 1e-6 meters between frames)
- Warning logs: "Sequence X: Insufficient GNSS data (avg step: Y m)"

**Causes:**
- Stationary camera or minimal camera movement
- GPS signal loss or poor quality
- Frames clustered in a very small area

**Pipeline Response:**
- Falls back to default scale factor of 1.0
- Logs warning messages identifying affected sequences
- Continues processing with synthetic scale
- May result in incorrect absolute dimensions

**Recovery Options:**
- Provide reference tracks with known scale
- Use ground control points (anchors) for scale estimation
- Filter out stationary sequences during preprocessing

### Reconstruction Failures

**Symptoms:**
- Missing reconstruction data for sequences
- Warning logs: "Sequences without valid reconstruction data: [...]"

**Causes:**
- Insufficient image overlap for SfM/VO
- Poor image quality (blur, low contrast, repetitive textures)
- Failed feature detection or matching
- Extreme camera motion

**Pipeline Response:**
- Attempts to use remaining valid reconstruction sources
- Falls back to default scale of 1.0 if no sources available
- Logs informative warnings about missing data
- Continues with available data

**Recovery Options:**
- Check reconstruction logs for detailed failure information
- Adjust SfM/VO parameters (feature detector, matcher settings)
- Improve image quality or capture strategy
- Add more images with better overlap

### Extreme Scale Discrepancies

**Symptoms:**
- Scale values clamped to bounds [0.25, 4.0]
- Warning logs: "Sequence X: Extreme scale Y from Z"
- Info logs: "Scale clamped from X to Y"

**Causes:**
- Significant GPS positioning errors (multipath, poor satellite geometry)
- Reconstruction drift or accumulated error
- Mixed coordinate systems or unit mismatches
- Camera calibration issues

**Pipeline Response:**
- Filters out extreme scale candidates (< 0.01 or > 100.0)
- Clamps final scale to reasonable range [0.25, 4.0]
- Logs both original and clamped values
- Uses remaining valid scale sources if available

**Recovery Options:**
- Inspect GPS quality metrics and filter poor-quality positions
- Check reconstruction coordinate frame consistency
- Verify camera calibration parameters
- Use multiple independent reconstruction methods for cross-validation

### Non-finite Values

**Symptoms:**
- Warning logs: "Non-finite GNSS step detected" or "Non-finite scale from X"

**Causes:**
- NaN or Inf values in GPS coordinates
- Division by zero in scale computation
- Numerical overflow in reconstruction

**Pipeline Response:**
- Detects and filters non-finite values at multiple stages
- Falls back to safe defaults (1.0 for scale, midpoint for height)
- Logs warnings with diagnostic information
- Continues processing with valid data

**Recovery Options:**
- Fix data quality issues in preprocessing
- Investigate source of invalid coordinates
- Check for sensor failures or data corruption

## Height Estimation Failures

### Missing Altitude Data

**Symptoms:**
- Warning logs: "No frames with altitude data, using default height"
- Height set to default midpoint (2.0m for typical range [1.0, 3.0])

**Causes:**
- GPS altitudes not available in frame metadata
- Barometric sensor failures
- Data export/import issues

**Pipeline Response:**
- Uses default height at midpoint of valid range
- Logs diagnostic messages
- Continues processing

**Recovery Options:**
- Check data source for altitude availability
- Use alternative elevation sources (DEM, anchors)
- Configure height range appropriately for use case

### No Valid Anchors

**Symptoms:**
- Debug logs: "No valid anchors, using default height"
- Height estimation falls back to GPS-only method

**Causes:**
- No ground control points detected
- Anchor detection failures
- Anchors outside sequence coverage area

**Pipeline Response:**
- Uses camera altitude from GPS as reference
- Falls back to default height if GPS altitude unavailable
- Continues with best available estimate

**Recovery Options:**
- Improve anchor detection sensitivity
- Add manual ground control points
- Verify anchor observations in images

## Consensus Voting Failures

### Insufficient Spatial Overlap

**Symptoms:**
- Empty consensus point set
- No cells with 2+ sources

**Causes:**
- Reconstruction sources cover different areas
- Poor alignment between coordinate systems
- Insufficient coverage density

**Pipeline Response:**
- Returns empty consensus set
- Downstream stages should handle gracefully
- May proceed with individual source data if available

**Recovery Options:**
- Verify coordinate frame consistency across sources
- Increase reconstruction coverage/density
- Adjust grid resolution (GRID_RES_M) to account for alignment uncertainty
- Check for systematic offsets between sources

### Height Disagreement Beyond Threshold

**Symptoms:**
- Fewer consensus points than expected
- Sources present in same area but not agreeing

**Causes:**
- Reconstruction height biases or systematic errors
- Scale estimation failures propagating
- Vertical datum mismatches
- Terrain complexity (vegetation, steep slopes)

**Pipeline Response:**
- Only creates consensus where agreement exists
- Filters out disagreeing measurements
- Preserves separate source data for inspection

**Recovery Options:**
- Tune DZ_MAX_M threshold based on empirical accuracy
- Investigate systematic height offsets between sources
- Check vertical datum and ellipsoid height handling
- Review scale estimation quality

### Single Source Coverage

**Symptoms:**
- No consensus points in areas with only one source
- Uneven coverage in final output

**Causes:**
- Gaps in one or more reconstruction sources
- Source-specific failures in certain areas

**Pipeline Response:**
- Requires minimum 2 sources for consensus
- Areas with single source are excluded
- Logs available for identifying coverage gaps

**Recovery Options:**
- Fill coverage gaps with additional data
- Consider using single-source data with appropriate uncertainty
- Adjust pipeline to handle mixed consensus/single-source regions

## Complete Pipeline Failure

### All Sequences Fail

**Symptoms:**
- ValueError: "Failed to compute scale for any sequence"

**Causes:**
- Systematic data quality issues
- Configuration errors
- All reconstruction methods failing
- Completely invalid GPS data

**Pipeline Response:**
- Raises ValueError with diagnostic message
- Suggests common causes and recovery steps
- Halts pipeline execution to prevent invalid outputs

**Recovery Options:**
1. Check input data quality and format
2. Verify reconstruction pipeline configuration
3. Review GPS data validity and coverage
4. Inspect reconstruction logs for systematic failures
5. Test with known-good reference dataset

## Monitoring and Diagnostics

### Log Levels

- **ERROR**: Critical failures requiring intervention
- **WARNING**: Issues that may affect quality but don't halt processing
- **INFO**: Informative messages about clamping, fallbacks, adjustments
- **DEBUG**: Detailed diagnostic information for troubleshooting

### Key Metrics to Monitor

- Percentage of sequences with valid scale estimation
- Number of sequences falling back to default scale
- Scale clamping frequency and magnitude
- Consensus point density vs. input point density
- Source agreement statistics (height RMSE between sources)

### Quality Assurance Checks

- Compare scale estimates across reconstruction methods
- Verify height estimates against known references
- Inspect consensus support counts and source combinations
- Monitor uncertainty distributions in consensus output
