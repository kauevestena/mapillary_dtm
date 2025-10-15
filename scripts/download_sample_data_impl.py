#!/usr/bin/env python3
"""
Download Sample Data for DTM from Mapillary Pipeline - Implementation

This module contains the actual implementation for downloading sample data.
It should be run via the launcher script (download_sample_data.py) to ensure
proper Python path configuration.
"""

import argparse
import json
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

# Set up Python path if not already done
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
os.chdir(str(_project_root))

# Import project modules - must import the package properly
import dtm_from_mapillary
from dtm_from_mapillary import constants
from dtm_from_mapillary.api.mapillary_client import MapillaryClient
from dtm_from_mapillary.ingest.sequence_scan import discover_sequences
from dtm_from_mapillary.ingest.sequence_filter import filter_car_sequences
from dtm_from_mapillary.ingest.imagery_cache import prefetch_imagery
from dtm_from_mapillary.common_core import FrameMeta

# Get default bbox from constants
DEFAULT_BBOX = constants.bbox

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


def create_directory_structure(base_dir: Path) -> None:
    """Create the sample dataset directory structure."""
    log.info(f"Creating directory structure in {base_dir}...")

    subdirs = [
        "imagery",  # Cached Mapillary thumbnails
        "sequences",  # Sequence metadata (JSONL)
        "reference_dtm",  # Optional reference DTM for validation
        "ground_truth",  # Optional ground truth checkpoints
        "outputs",  # Pipeline outputs
        "qa_reports",  # QA reports and visualizations
    ]

    for subdir in subdirs:
        (base_dir / subdir).mkdir(parents=True, exist_ok=True)
        log.info(f"  ✓ Created {subdir}/")


def create_dataset_readme(
    base_dir: Path, bbox: Dict[str, float], stats: Dict[str, Any]
) -> None:
    """Create a README for the specific dataset."""

    bbox_str = f"{bbox['min_lon']:.6f},{bbox['min_lat']:.6f},{bbox['max_lon']:.6f},{bbox['max_lat']:.6f}"

    readme_content = f"""# Mapillary DTM Sample Dataset

**Location:** Florianópolis, SC, Brazil (default from constants.py)  
**Bounding Box:** `{bbox_str}`  
**Downloaded:** {datetime.now().isoformat()}

## Dataset Statistics

- **Total Sequences:** {stats.get('total_sequences', 0)}
- **Car-Only Sequences:** {stats.get('car_sequences', 0)}
- **Total Images:** {stats.get('total_images', 0)}
- **Cached Thumbnails:** {stats.get('cached_images', 0)}

## Directory Structure

```
{base_dir.name}/
├── imagery/              # Cached Mapillary thumbnails (1024px)
├── sequences/            # Sequence metadata (JSONL format)
├── reference_dtm/        # Optional reference DTM for validation
├── ground_truth/         # Optional ground truth checkpoints
├── outputs/              # Pipeline outputs (DTM, slope maps, etc.)
├── qa_reports/           # QA reports and visualizations
└── README.md            # This file
```

## Data Sources

### Mapillary
- **API Version:** Graph API v4
- **Data Type:** Street-level imagery sequences (car-mounted cameras only)
- **Coverage:** Sequences cached in `cache/mapillary/metadata/`
- **Imagery:** Thumbnails cached in `cache/mapillary/imagery/` (if downloaded)

### Reference DTM (Optional)
To enable external validation, place a reference DTM in `reference_dtm/`:

**Recommended sources:**
- **USGS 3DEP:** https://apps.nationalmap.gov/downloader/
- **OpenTopography:** https://opentopography.org/
- **Brazilian TOPODATA:** http://www.dsr.inpe.br/topodata/
- **State/County GIS portals**

**Requirements:**
- Format: GeoTIFF
- CRS: Any (will be reprojected automatically)
- Heights: Preferably orthometric (geoid-based) or ellipsoidal

## Usage

### 1. Run Pipeline on This Dataset

```bash
# Basic run (ellipsoidal heights, 0.5m resolution)
python -m cli.pipeline run \\
  --aoi-bbox "{bbox_str}" \\
  --out-dir {base_dir}/outputs

# With breakline enforcement (curbs/edges)
python -m cli.pipeline run \\
  --aoi-bbox "{bbox_str}" \\
  --out-dir {base_dir}/outputs \\
  --enforce-breaklines

# With learned uncertainty calibration
python -m cli.pipeline run \\
  --aoi-bbox "{bbox_str}" \\
  --out-dir {base_dir}/outputs \\
  --use-learned-uncertainty
```

### 2. QA Validation (after obtaining reference DTM)

```bash
python -m qa.qa_external \\
  --generated-dtm {base_dir}/outputs/dtm_0p5m_ellipsoid.tif \\
  --reference-dtm {base_dir}/reference_dtm/reference.tif \\
  --report-out {base_dir}/qa_reports/external_validation.html
```

### 3. Convert to Orthometric Heights (optional)

```bash
python -m io.geoutils convert_to_orthometric \\
  --input-dtm {base_dir}/outputs/dtm_0p5m_ellipsoid.tif \\
  --output-dtm {base_dir}/outputs/dtm_0p5m_orthometric.tif \\
  --geoid-model EGM96
```

## Sequence Details

Sequences are filtered to car-mounted cameras only (40-120 km/h speed analysis).
See `sequences/filtered_sequences.json` for details.

## Notes

- **Camera Height Range:** 1.0 - 3.0 meters (typical car-mounted cameras)
- **Resolution:** 0.5m grid resolution (configurable in constants.py)
- **Coordinate System:** Output DTMs use ellipsoidal heights (EPSG:4979)
- **Corridor Buffer:** 25m around OSM street centerlines (configurable)

## Troubleshooting

### No sequences found
- Check that `MAPILLARY_TOKEN` environment variable is set
- Verify bbox overlaps with Mapillary coverage: https://www.mapillary.com/app/
- Try expanding the bbox area

### API rate limits
- Sequences and metadata are cached in `cache/mapillary/`
- Use `--force-refresh` only when necessary
- Consider running during off-peak hours

### Missing imagery
- Thumbnails are only downloaded if `--cache-imagery` flag is used
- Pipeline will fetch imagery on-demand during execution if not cached

---

*Generated by download_sample_data.py*  
*See docs/ROADMAP.md for pipeline details*
"""

    with open(base_dir / "README.md", "w") as f:
        f.write(readme_content)

    log.info("✓ Created dataset README")


def create_config_json(
    base_dir: Path, bbox: Dict[str, float], stats: Dict[str, Any]
) -> None:
    """Create a JSON config file for the dataset."""

    config = {
        "name": "Florianópolis Sample Dataset",
        "location": "Florianópolis, SC, Brazil",
        "bbox": bbox,
        "bbox_string": f"{bbox['min_lon']},{bbox['min_lat']},{bbox['max_lon']},{bbox['max_lat']}",
        "statistics": stats,
        "created": datetime.now().isoformat(),
        "pipeline_version": "1.0",
        "data_source": "Mapillary Graph API v4",
        "notes": "Sample dataset for DTM generation pipeline testing and validation",
    }

    with open(base_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    log.info("✓ Created config.json")


def save_sequence_stats(
    base_dir: Path,
    sequences: Dict[str, List[FrameMeta]],
    filtered_sequences: Dict[str, List[FrameMeta]],
) -> None:
    """Save sequence statistics to JSON."""

    stats = {
        "total_sequences_discovered": len(sequences),
        "car_only_sequences": len(filtered_sequences),
        "total_images": sum(len(frames) for frames in sequences.values()),
        "car_images": sum(len(frames) for frames in filtered_sequences.values()),
        "sequence_details": {},
    }

    for seq_id, frames in filtered_sequences.items():
        if not frames:
            continue
        stats["sequence_details"][seq_id] = {
            "frame_count": len(frames),
            "first_captured": frames[0].captured_at_ms,
            "last_captured": frames[-1].captured_at_ms,
            "camera_type": (
                frames[0].camera_type if frames[0].camera_type else "unknown"
            ),
        }

    with open(base_dir / "sequences" / "filtered_sequences.json", "w") as f:
        json.dump(stats, f, indent=2)

    log.info("✓ Saved sequence statistics")


def main():
    parser = argparse.ArgumentParser(
        description="Download Mapillary sample data for DTM pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default bbox from constants.py
  python scripts/download_sample_data.py
  
  # Custom bbox
  python scripts/download_sample_data.py --bbox "-48.6,-27.6,-48.59,-27.59"
  
  # Download more images per sequence
  python scripts/download_sample_data.py --images-per-sequence 10
  
  # Force refresh (ignore cache)
  python scripts/download_sample_data.py --force-refresh

Environment:
  MAPILLARY_TOKEN    Required. Your Mapillary API token.
                     Get one at: https://www.mapillary.com/dashboard/developers
""",
    )

    parser.add_argument(
        "--bbox",
        help="Custom bounding box (lon_min,lat_min,lon_max,lat_max). "
        "Default: bbox from constants.py (Florianópolis, Brazil)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/sample_dataset"),
        help="Output directory (default: data/sample_dataset)",
    )

    parser.add_argument(
        "--cache-imagery",
        action="store_true",
        help="Download and cache Mapillary thumbnail images (1024px)",
    )

    parser.add_argument(
        "--images-per-sequence",
        type=int,
        default=5,
        help="Maximum thumbnails to cache per sequence (default: 5)",
    )

    parser.add_argument(
        "--max-sequences",
        type=int,
        help="Limit number of sequences to download (for testing)",
    )

    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh (ignore cached metadata)",
    )

    parser.add_argument(
        "--token", help="Mapillary API token (or set MAPILLARY_TOKEN env var)"
    )

    args = parser.parse_args()

    # Parse bbox
    if args.bbox:
        try:
            coords = [float(x.strip()) for x in args.bbox.split(",")]
            if len(coords) != 4:
                raise ValueError("Bbox must have 4 coordinates")
            bbox_dict = {
                "min_lon": coords[0],
                "min_lat": coords[1],
                "max_lon": coords[2],
                "max_lat": coords[3],
            }
            bbox_tuple = tuple(coords)
        except (ValueError, IndexError) as e:
            parser.error(f"Invalid bbox format: {e}")
    else:
        bbox_dict = DEFAULT_BBOX
        bbox_tuple = (
            bbox_dict["min_lon"],
            bbox_dict["min_lat"],
            bbox_dict["max_lon"],
            bbox_dict["max_lat"],
        )

    bbox_str = f"{bbox_tuple[0]:.6f},{bbox_tuple[1]:.6f},{bbox_tuple[2]:.6f},{bbox_tuple[3]:.6f}"

    print("=" * 70)
    print("Mapillary DTM Sample Data Download")
    print("=" * 70)
    print(f"Bounding box: {bbox_str}")
    print(f"Output directory: {args.output_dir}")
    print(f"Cache imagery: {'Yes' if args.cache_imagery else 'No'}")
    if args.max_sequences:
        print(f"Max sequences: {args.max_sequences}")
    print("=" * 70)
    print()

    # Create directory structure
    create_directory_structure(args.output_dir)
    print()

    # Initialize Mapillary client
    try:
        log.info("Initializing Mapillary client...")
        client = MapillaryClient(token=args.token)
        log.info("✓ Mapillary client initialized")
    except RuntimeError as e:
        log.error(f"Failed to initialize Mapillary client: {e}")
        print("\n❌ Error: Mapillary token not found!")
        print("Set MAPILLARY_TOKEN environment variable or use --token flag")
        print("Get a token at: https://www.mapillary.com/dashboard/developers")
        sys.exit(1)

    print()

    # Discover sequences
    try:
        log.info("Discovering sequences in bounding box...")
        sequences = discover_sequences(
            aoi_bbox=bbox_tuple,
            client=client,
            max_sequences=args.max_sequences,
            use_cache=not args.force_refresh,
            force_refresh=args.force_refresh,
        )
        log.info(f"✓ Discovered {len(sequences)} sequences")

        if not sequences:
            log.warning("⚠️  No sequences found in this area!")
            log.warning("Check Mapillary coverage: https://www.mapillary.com/app/")
            print("\n" + "=" * 70)
            print("No sequences found. Try:")
            print("  1. Expanding the bounding box")
            print("  2. Checking coverage at https://www.mapillary.com/app/")
            print("  3. Using a different area")
            print("=" * 70)
            sys.exit(1)

    except Exception as e:
        log.error(f"Failed to discover sequences: {e}", exc_info=True)
        sys.exit(1)

    print()

    # Filter to car-only sequences
    try:
        log.info("Filtering car-only sequences (40-120 km/h speed analysis)...")
        filtered_sequences = filter_car_sequences(sequences)
        log.info(f"✓ Retained {len(filtered_sequences)} car-only sequences")

        if not filtered_sequences:
            log.warning("⚠️  No car sequences found after filtering!")
            log.warning("This area may only have pedestrian/bike imagery")
    except Exception as e:
        log.error(f"Failed to filter sequences: {e}", exc_info=True)
        filtered_sequences = sequences  # Fallback to all sequences

    print()

    # Save sequence statistics
    try:
        save_sequence_stats(args.output_dir, sequences, filtered_sequences)
    except Exception as e:
        log.warning(f"Failed to save sequence stats: {e}")

    # Cache imagery if requested
    cached_images = 0
    if args.cache_imagery and filtered_sequences:
        try:
            log.info(f"Caching up to {args.images_per_sequence} images per sequence...")
            cache_stats = prefetch_imagery(
                filtered_sequences,
                client=client,
                max_per_sequence=args.images_per_sequence,
            )
            cached_images = sum(cache_stats.values()) if cache_stats else 0
            log.info(f"✓ Cached {cached_images} thumbnail images")
        except Exception as e:
            log.error(f"Failed to cache imagery: {e}", exc_info=True)

    print()

    # Create README and config
    stats = {
        "total_sequences": len(sequences),
        "car_sequences": len(filtered_sequences),
        "total_images": sum(len(frames) for frames in sequences.values()),
        "cached_images": cached_images,
    }

    try:
        create_dataset_readme(args.output_dir, bbox_dict, stats)
        create_config_json(args.output_dir, bbox_dict, stats)
    except Exception as e:
        log.warning(f"Failed to create documentation: {e}")

    print()
    print("=" * 70)
    print("✓ Sample dataset download complete!")
    print()
    print("Summary:")
    print(f"  • Total sequences discovered: {stats['total_sequences']}")
    print(f"  • Car-only sequences: {stats['car_sequences']}")
    print(f"  • Total images: {stats['total_images']}")
    if args.cache_imagery:
        print(f"  • Cached thumbnails: {stats['cached_images']}")
    print()
    print(f"Dataset location: {args.output_dir.absolute()}")
    print()
    print("Next steps:")
    print("  1. (Optional) Add reference DTM to reference_dtm/ for validation")
    print("  2. Run the pipeline:")
    print(f"     python -m cli.pipeline run \\")
    print(f'       --aoi-bbox "{bbox_str}" \\')
    print(f"       --out-dir {args.output_dir}/outputs")
    print()
    print("See data/sample_dataset/README.md for detailed usage instructions")
    print("=" * 70)


if __name__ == "__main__":
    main()
