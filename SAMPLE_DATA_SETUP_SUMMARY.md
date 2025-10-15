# Sample Data Download Setup - Summary

## ✅ What Was Done

### 1. `.gitignore` Updated
- Added `data/` directory to gitignore
- Data directory (with large Mapillary downloads) will not be committed to git
- Keeps repository clean and lightweight

### 2. Download Script Created
- **Main script:** `scripts/download_sample_data.py`
- **Implementation:** `scripts/download_sample_data_impl.py`
- **Shell wrapper:** `scripts/download_sample_data.sh`
- Downloads actual Mapillary data from bbox in `constants.py`

### 3. Documentation Added
- **[SAMPLE_DATA_GUIDE.md](SAMPLE_DATA_GUIDE.md)** - Complete usage guide
- **[data/README.md](data/README.md)** - Data directory documentation
- Both files provide comprehensive instructions

### 4. Directory Structure
The script will create:
```
data/sample_dataset/
├── README.md              # Auto-generated dataset documentation
├── config.json            # Configuration and statistics
├── imagery/               # Cached Mapillary thumbnails
├── sequences/             # Sequence metadata
├── reference_dtm/         # (Manual) Reference DTM for validation
├── ground_truth/          # (Manual) Optional ground truth
├── outputs/               # Pipeline outputs
└── qa_reports/            # QA validation reports
```

---

## 🚀 How to Use

### Quick Start (3 steps):

```bash
# 1. Set your Mapillary token
export MAPILLARY_TOKEN="your_token_here"

# 2. Download sample data from Florianópolis, Brazil (default in constants.py)
.venv/bin/python scripts/download_sample_data.py --cache-imagery --images-per-sequence 10

# 3. Run the pipeline
python -m dtm_from_mapillary.cli.pipeline run \
  --aoi-bbox "-48.596644,-27.591363,-48.589890,-27.586780" \
  --out-dir data/sample_dataset/outputs
```

---

## 📍 Default Sample Area

The script uses the bbox from `constants.py`:

```python
bbox = {
    "min_lon": -48.596644,
    "min_lat": -27.591363,
    "max_lon": -48.589890,
    "max_lat": -27.586780,
}  # Florianópolis, SC, Brazil
```

**Characteristics:**
- ~700m × 500m area
- Urban coastal streets
- Moderate slopes
- Expected: 5-15 car sequences

---

## 🎯 Script Features

### Downloads from Mapillary
- ✅ Discovers sequences in bbox
- ✅ Filters car-only sequences (40-120 km/h)
- ✅ Caches metadata (JSONL)
- ✅ Optionally caches imagery (thumbnails)
- ✅ Generates documentation

### Outputs
- Sequence statistics JSON
- Dataset README with usage instructions
- Configuration file with bbox and stats

### Smart Caching
- Uses Mapillary cache system
- Respects rate limits
- Supports force-refresh

---

## 📦 What Gets Downloaded

### Metadata (Always, ~5-10 MB)
- Sequence IDs and properties
- Frame metadata (position, camera, quality)
- Stored in: `cache/mapillary/metadata/`

### Imagery (Optional, ~50-200 MB)
- Mapillary thumbnails (1024px)
- Controlled by `--images-per-sequence`
- Stored in: `cache/mapillary/imagery/`

---

## 🔧 Script Options

```
--bbox BBOX                     Custom bounding box
--output-dir DIR                Output directory (default: data/sample_dataset)
--cache-imagery                 Download thumbnail images
--images-per-sequence N         Max thumbnails per sequence (default: 5)
--max-sequences N               Limit sequences (for testing)
--force-refresh                 Ignore cache
--token TOKEN                   Mapillary API token
```

---

## 📖 Documentation Files

1. **[SAMPLE_DATA_GUIDE.md](SAMPLE_DATA_GUIDE.md)**
   - Complete usage guide
   - Troubleshooting
   - Next steps

2. **[data/README.md](data/README.md)**
   - Data directory structure
   - Manual download instructions
   - Reference DTM sources

3. **data/sample_dataset/README.md** (auto-generated)
   - Dataset-specific documentation
   - Pipeline commands for this dataset
   - Statistics and metadata

---

## 🛠️ Technical Details

### Import Structure
- Script uses launcher pattern to handle Python package imports
- `download_sample_data.py` - Launcher (sets up Python path)
- `download_sample_data_impl.py` - Implementation (actual logic)

### Dependencies
- Uses existing pipeline modules:
  - `api.mapillary_client.MapillaryClient`
  - `ingest.sequence_scan.discover_sequences`
  - `ingest.sequence_filter.filter_car_sequences`
  - `ingest.imagery_cache.prefetch_imagery`

### Caching
- Leverages existing Mapillary cache system
- Respects `MAPILLARY_CACHE_ROOT` from constants
- Sequences cached as JSONL files
- Imagery cached by image ID

---

## ✅ Verification

Test the script:

```bash
# Show help
.venv/bin/python scripts/download_sample_data.py --help

# Test with small download (limit sequences)
.venv/bin/python scripts/download_sample_data.py \
  --max-sequences 2 \
  --images-per-sequence 3
```

Expected output:
```
=============================================================================
Mapillary DTM Sample Data Download
=============================================================================
Bounding box: -48.596644,-27.591363,-48.589890,-27.586780
Output directory: data/sample_dataset
Cache imagery: No
=============================================================================

[Logs about discovering sequences, filtering, etc.]

=============================================================================
✓ Sample dataset download complete!

Summary:
  • Total sequences discovered: X
  • Car-only sequences: Y
  • Total images: Z

Dataset location: /path/to/data/sample_dataset
...
```

---

## 🔜 Next Steps

1. **Set Mapillary Token**
   ```bash
   export MAPILLARY_TOKEN="your_token"
   ```

2. **Run Download**
   ```bash
   .venv/bin/python scripts/download_sample_data.py --cache-imagery
   ```

3. **Verify Data**
   ```bash
   ls -lh data/sample_dataset/
   cat data/sample_dataset/config.json
   ```

4. **Run Pipeline**
   ```bash
   python -m dtm_from_mapillary.cli.pipeline run \
     --aoi-bbox "-48.596644,-27.591363,-48.589890,-27.586780" \
     --out-dir data/sample_dataset/outputs
   ```

5. **(Optional) Add Reference DTM**
   - Download from TOPODATA: http://www.dsr.inpe.br/topodata/
   - Place in: `data/sample_dataset/reference_dtm/reference.tif`

6. **(Optional) Run QA Validation**
   ```bash
   python -m dtm_from_mapillary.qa.qa_external \
     --generated-dtm data/sample_dataset/outputs/dtm_0p5m_ellipsoid.tif \
     --reference-dtm data/sample_dataset/reference_dtm/reference.tif \
     --report-out data/sample_dataset/qa_reports/validation.html
   ```

---

## 📚 Related Documentation

- **[agents.md](agents.md)** - AI agent guide (project overview)
- **[docs/ROADMAP.md](docs/ROADMAP.md)** - Implementation roadmap
- **[README.md](README.md)** - Project README
- **[constants.py](constants.py)** - Configuration (including bbox)

---

*Setup completed: October 14, 2025*
*Script location: scripts/download_sample_data.py*
