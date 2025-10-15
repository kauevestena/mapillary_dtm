# QA Data Directory

This directory contains ground truth Digital Terrain Model (DTM) files used for quality assurance and validation.

## Files

### `qa_dtm.tif`
Ground Truth DTM with geoidal reference (vertical datum).
- Contains elevation data referenced to the geoid
- Uses the original coordinate reference system (CRS)

### `qa_dtm_4326.tif`
Same ground truth DTM data as `qa_dtm.tif`, but reprojected for comparison purposes.
- **Vertical datum**: Same geoidal reference as the original
- **Horizontal CRS**: EPSG:4326 (WGS84 geographic coordinates)
- This version facilitates comparisons with data in geographic coordinates

### `opensfm_fixture/`
Reference reconstruction exported from OpenSfM for adapter regression tests.
- Contains a minimal `reconstruction.json` bundle used by `geom/opensfm_adapter.py`
- Enables fixture-driven runs without invoking the real OpenSfM binary

## Purpose

These files serve as reference datasets for:
- Validating DTM generation outputs
- Running quality assurance tests
- Comparing generated terrain models against known ground truth data
