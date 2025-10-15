# Sample Data Download - Quick Start Guide

## ✅ Setup Complete!

The sample data download script is now ready to use. Here's everything you need to know:

---

## 📋 What's Been Set Up

1. **`.gitignore` updated** - `data/` directory is now excluded from git
2. **Download script created** - `scripts/download_sample_data.py` downloads Mapillary data
3. **Directory structure** - `data/` will be created with proper subdirectories
4. **Documentation** - README files explain usage and structure

---

## 🚀 Quick Start

### Step 1: Set Your Mapillary Token

```bash
export MAPILLARY_TOKEN="your_token_here"
```

Get a free token at: https://www.mapillary.com/dashboard/developers

### Step 2: Download Sample Data

From the project root directory:

```bash
# Basic download (metadata only) - uses bbox from constants.py
.venv/bin/python scripts/download_sample_data.py

# Recommended: Download with imagery caching
.venv/bin/python scripts/download_sample_data.py --cache-imagery --images-per-sequence 10

# Custom area
.venv/bin/python scripts/download_sample_data.py --bbox "-48.6,-27.6,-48.59,-27.59" --cache-imagery
```

### Step 3: Run the Pipeline

```bash
python -m dtm_from_mapillary.cli.pipeline run \
  --aoi-bbox "-48.596644,-27.591363,-48.589890,-27.586780" \
  --out-dir data/sample_dataset/outputs \
  --cache-imagery
```

---

## 📁 Default Sample Area

The script uses the bbox defined in `constants.py` by default:

- **Location:** Florianópolis, SC, Brazil
- **Coordinates:** 
  - Min: -48.596644°, -27.591363°
  - Max: -48.589890°, -27.586780°
- **Size:** ~700m × 500m urban coastal area
- **Expected:** 5-15 car sequences

---

## 🎯 Script Options

```bash
.venv/bin/python scripts/download_sample_data.py [OPTIONS]

Options:
  --bbox BBOX                     Custom bounding box (lon_min,lat_min,lon_max,lat_max)
  --output-dir DIR                Output directory (default: data/sample_dataset)
  --cache-imagery                 Download thumbnail images (1024px)
  --images-per-sequence N         Max thumbnails per sequence (default: 5)
  --max-sequences N               Limit number of sequences (for testing)
  --force-refresh                 Ignore cached metadata
  --token TOKEN                   Mapillary API token (or use env var)
  -h, --help                      Show help message
```

---

## 📊 What Gets Downloaded

### Metadata (Always)
- Sequence information (camera type, capture date, GNSS positions)
- Frame metadata (timestamps, camera parameters, quality scores)
- Stored in: `cache/mapillary/metadata/` (JSONL format)

### Imagery (Optional, with `--cache-imagery`)
- Mapillary thumbnails (1024px resolution)
- Limited by `--images-per-sequence` to control disk usage
- Stored in: `cache/mapillary/imagery/`

### Generated Files
- `data/sample_dataset/README.md` - Dataset documentation
- `data/sample_dataset/config.json` - Configuration and statistics
- `data/sample_dataset/sequences/filtered_sequences.json` - Sequence details

---

## 💾 Disk Space

Typical usage for default bbox:

- **Metadata only:** ~5-10 MB
- **With imagery (5 images/sequence):** ~50-100 MB
- **With imagery (10 images/sequence):** ~100-200 MB
- **Pipeline outputs:** ~10-50 MB (DTM, slope maps, LAZ)

**Total:** 100-300 MB for a complete test dataset

---

## 🔍 Directory Structure Created

```
data/sample_dataset/
├── README.md              # Dataset documentation
├── config.json            # Configuration and stats
├── imagery/               # Cached thumbnails (if --cache-imagery used)
├── sequences/             # Sequence statistics
│   └── filtered_sequences.json
├── reference_dtm/         # (Manual) Place reference DTM here for validation
├── ground_truth/          # (Manual) Optional ground truth points
├── outputs/               # Pipeline outputs (after running pipeline)
│   ├── dtm_0p5m_ellipsoid.tif
│   ├── slope_deg.tif
│   ├── slope_pct.tif
│   └── ground_points.laz
└── qa_reports/            # QA reports (after validation)
    └── external_validation.html
```

---

## 🛠️ Troubleshooting

### "Mapillary token not found"
- Set `MAPILLARY_TOKEN` environment variable
- Or use `--token` flag
- Get token at: https://www.mapillary.com/dashboard/developers

### "No sequences found"
- Check coverage at: https://www.mapillary.com/app/
- Try expanding the bbox
- Verify bbox format: `lon_min,lat_min,lon_max,lat_max`

### "Import errors"
- Ensure you're running from project root
- Use `.venv/bin/python` (not system python)
- Check that virtual environment is set up: `.venv/bin/pip install -r requirements.txt`

### "API rate limit"
- Sequences are cached automatically in `cache/mapillary/`
- Don't use `--force-refresh` unnecessarily
- Wait a few minutes and retry

---

## 🎓 Next Steps

After downloading sample data:

### 1. Add Reference DTM (Optional)

For external validation, get a reference DTM:

**For Brazil (Florianópolis):**
- TOPODATA (INPE): http://www.dsr.inpe.br/topodata/
- 30m SRTM-derived DTM

Place in: `data/sample_dataset/reference_dtm/reference.tif`

### 2. Run Pipeline

```bash
python -m dtm_from_mapillary.cli.pipeline run \
  --aoi-bbox "-48.596644,-27.591363,-48.589890,-27.586780" \
  --out-dir data/sample_dataset/outputs
```

### 3. Quality Assurance (if you have reference DTM)

```bash
python -m dtm_from_mapillary.qa.qa_external \
  --generated-dtm data/sample_dataset/outputs/dtm_0p5m_ellipsoid.tif \
  --reference-dtm data/sample_dataset/reference_dtm/reference.tif \
  --report-out data/sample_dataset/qa_reports/external_validation.html
```

---

## 📖 Additional Documentation

- **[data/README.md](../data/README.md)** - Data directory documentation
- **[docs/ROADMAP.md](../docs/ROADMAP.md)** - Pipeline implementation details
- **[README.md](../README.md)** - Project overview
- **[agents.md](../agents.md)** - AI agent guide

---

## 🔗 Useful Links

- **Mapillary Developer Portal:** https://www.mapillary.com/dashboard/developers
- **Mapillary Coverage Map:** https://www.mapillary.com/app/
- **TOPODATA (Brazil DTM):** http://www.dsr.inpe.br/topodata/
- **OpenTopography (Global DTM):** https://opentopography.org/
- **USGS 3DEP (US DTM):** https://apps.nationalmap.gov/downloader/

---

*Last Updated: October 14, 2025*
*Script: scripts/download_sample_data.py*
