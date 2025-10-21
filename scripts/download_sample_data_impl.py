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
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

import requests
from tqdm import tqdm

# Set up Python path if not already done
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
os.chdir(str(_project_root))

# Import project modules - must import the package properly
import dtm_from_mapillary
from dtm_from_mapillary import constants
from dtm_from_mapillary.common_core import FrameMeta
from dtm_from_mapillary.ingest.sequence_filter import filter_car_sequences

_mapillary_api_dir = _project_root / "api" / "my_mapillary_api"
if str(_mapillary_api_dir) not in sys.path:
    sys.path.insert(0, str(_mapillary_api_dir))

import mapillary_api  # type: ignore[import]

# Get default bbox from constants
DEFAULT_BBOX = constants.bbox

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


def resolve_token(explicit: Optional[str]) -> str:
    """Resolve Mapillary token from CLI, env vars, or token file."""
    if explicit and explicit.strip():
        return explicit.strip()

    env_token = os.getenv("MAPILLARY_TOKEN")
    if env_token and env_token.strip():
        return env_token.strip()

    token = mapillary_api.get_mapillary_token()
    if token and token.strip():
        return token.strip()

    raise RuntimeError(
        "Mapillary token not provided. "
        "Use --token, export MAPILLARY_TOKEN, or create a mapillary_token file."
    )


def fetch_metadata_gdf(bbox: Dict[str, float], token: str, limit: Optional[int] = None):
    """Fetch Mapillary metadata within *bbox* using the simple demo logic."""
    max_items = limit or 5000
    data = mapillary_api.get_mapillary_images_metadata(
        bbox["min_lon"],
        bbox["min_lat"],
        bbox["max_lon"],
        bbox["max_lat"],
        token=token,
        limit=max_items,
    )
    return mapillary_api.mapillary_data_to_gdf(data)


def _captured_at_to_ms(raw) -> int:
    if raw is None:
        return 0
    if isinstance(raw, (int, float)):
        return int(raw)
    if isinstance(raw, str) and raw:
        try:
            if raw.endswith("Z"):
                raw_dt = raw[:-1] + "+00:00"
            else:
                raw_dt = raw
            dt = datetime.fromisoformat(raw_dt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000.0)
        except ValueError:
            return 0
    return 0


def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_cam_params(raw) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


def gdf_to_sequences(
    gdf, max_sequences: Optional[int] = None
) -> Dict[str, List[FrameMeta]]:
    """Convert GeoDataFrame rows to FrameMeta grouped by sequence."""
    if gdf is None or gdf.empty:
        return {}

    sequences: Dict[str, List[FrameMeta]] = {}
    seq_count = 0

    for row in gdf.itertuples(index=False):
        seq_id = getattr(row, "sequence", None)
        if not seq_id:
            continue
        if max_sequences is not None and seq_id not in sequences:
            if seq_count >= max_sequences:
                continue
            seq_count += 1

        geom = getattr(row, "geometry", None)
        lon = getattr(geom, "x", None) if geom is not None else None
        lat = getattr(geom, "y", None) if geom is not None else None
        if lon is None or lat is None:
            continue

        image_id = getattr(row, "id", None)
        if image_id is None:
            continue

        frame = FrameMeta(
            image_id=str(image_id),
            seq_id=str(seq_id),
            captured_at_ms=_captured_at_to_ms(getattr(row, "captured_at", None)),
            lon=float(lon),
            lat=float(lat),
            alt_ellip=_safe_float(getattr(row, "altitude", None)),
            camera_type=str(getattr(row, "camera_type", "unknown") or "unknown"),
            cam_params=_parse_cam_params(getattr(row, "camera_parameters", {})),
            quality_score=_safe_float(getattr(row, "quality_score", None)),
            thumbnail_url=getattr(row, "thumb_original_url", None),
        )

        sequences.setdefault(frame.seq_id, []).append(frame)

    for frames in sequences.values():
        frames.sort(key=lambda f: f.captured_at_ms)

    return sequences


def write_raw_metadata(gdf, out_path: Path) -> None:
    """Persist raw metadata to disk for inspection."""
    if gdf is None or gdf.empty:
        return
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(out_path, driver="GeoJSON")
        log.info("✓ Wrote raw metadata to %s", out_path)
    except Exception as exc:  # geopandas might not be available with fiona
        log.warning("Failed to write GeoJSON metadata (%s). Writing CSV fallback.", exc)
        try:
            gdf.to_csv(out_path.with_suffix(".csv"), index=False)
            log.info("✓ Wrote raw metadata CSV to %s", out_path.with_suffix(".csv"))
        except Exception as csv_exc:
            log.warning("Failed to write metadata CSV: %s", csv_exc)


def _download_bytes(url: str, dest_path: Path) -> None:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with dest_path.open("wb") as fh:
        fh.write(response.content)


def download_imagery_for_sequences(
    gdf, base_dir: Path, per_sequence: int | None
) -> Dict[str, int]:
    """
    Download thumbnails for each sequence with global progress tracking.

    Args:
        gdf: GeoDataFrame with image metadata
        base_dir: Directory to save images
        per_sequence: Max images per sequence (None = all images)

    Returns:
        Dictionary mapping sequence_id to number of downloaded images
    """
    if gdf is None or gdf.empty:
        return {}

    # If per_sequence is 0 or negative, download nothing
    if per_sequence is not None and per_sequence <= 0:
        return {}

    stats: Dict[str, int] = {}
    base_dir.mkdir(parents=True, exist_ok=True)

    if "sequence" not in gdf.columns:
        log.warning("Metadata is missing 'sequence' column; skipping imagery download.")
        return stats

    # Calculate total number of images to download
    grouped = gdf.groupby("sequence")
    total_images = 0
    images_already_cached = 0

    for seq_id, group in grouped:
        if not seq_id:
            continue
        images_to_download = group if per_sequence is None else group.head(per_sequence)

        for row in images_to_download.itertuples(index=False):
            url = getattr(row, "thumb_original_url", None)
            image_id = getattr(row, "id", None)
            if not url or image_id is None:
                continue
            dest_path = base_dir / str(seq_id) / f"{image_id}.jpg"
            if dest_path.exists():
                images_already_cached += 1
            else:
                total_images += 1

    if images_already_cached > 0:
        log.info(f"Found {images_already_cached} already cached images (skipping)")

    if total_images == 0:
        log.info("All images already cached!")
        # Count existing images for stats
        for seq_id, group in grouped:
            if not seq_id:
                continue
            images_to_count = (
                group if per_sequence is None else group.head(per_sequence)
            )
            count = 0
            for row in images_to_count.itertuples(index=False):
                image_id = getattr(row, "id", None)
                if image_id:
                    dest_path = base_dir / str(seq_id) / f"{image_id}.jpg"
                    if dest_path.exists():
                        count += 1
            if count > 0:
                stats[str(seq_id)] = count
        return stats

    log.info(f"Downloading {total_images} images...")

    # Download with global progress bar
    with tqdm(total=total_images, desc="Downloading images", unit="img") as pbar:
        for seq_id, group in grouped:
            if not seq_id:
                continue
            downloaded = 0

            # Select images: either all or limited by per_sequence
            images_to_download = (
                group if per_sequence is None else group.head(per_sequence)
            )

            for row in images_to_download.itertuples(index=False):
                url = getattr(row, "thumb_original_url", None)
                image_id = getattr(row, "id", None)
                if not url or image_id is None:
                    continue
                dest_path = base_dir / str(seq_id) / f"{image_id}.jpg"
                if dest_path.exists():
                    downloaded += 1
                    continue
                try:
                    _download_bytes(url, dest_path)
                    downloaded += 1
                    pbar.update(1)
                except requests.HTTPError as exc:
                    log.warning(
                        "Failed to download thumbnail for %s: %s", image_id, exc
                    )
                    pbar.update(1)
                except Exception as exc:
                    log.warning("Unexpected error downloading %s: %s", image_id, exc)
                    pbar.update(1)
            if downloaded:
                stats[str(seq_id)] = downloaded

    return stats


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

Sequences are filtered to car-mounted cameras only (30-120 km/h max-speed analysis).

- **Kept sequences:** See `sequences/filtered_sequences.json` with speed statistics
- **Rejected sequences:** See `sequences/filtered_out_sequences.json` for debugging filter behavior

The 30 km/h minimum threshold captures slower urban driving (residential streets, traffic)
while excluding pedestrian/bicycle sequences (typically < 25 km/h).

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


def save_filtered_sequences(
    base_dir: Path,
    filtered_sequences: dict[str, list],
    total_sequences: int,
    total_images: int,
    speed_stats: dict[str, dict] | None = None,
):
    """Save filtered sequences metadata with speed statistics"""
    stats = {
        "total_sequences_discovered": total_sequences,
        "car_only_sequences": len(filtered_sequences),
        "total_images": total_images,
        "car_images": sum(len(frames) for frames in filtered_sequences.values()),
        "sequence_details": {},
    }

    for seq_id, frames in filtered_sequences.items():
        if not frames:
            continue
        detail = {
            "frame_count": len(frames),
            "first_captured": frames[0].captured_at_ms,
            "last_captured": frames[-1].captured_at_ms,
            "camera_type": (
                frames[0].camera_type if frames[0].camera_type else "unknown"
            ),
        }

        # Add speed statistics if available
        if speed_stats and seq_id in speed_stats:
            speed_stat = speed_stats[seq_id]
            # Convert NamedTuple to dict if needed
            if hasattr(speed_stat, "_asdict"):
                detail["speed_statistics_kmh"] = speed_stat._asdict()
            else:
                detail["speed_statistics_kmh"] = speed_stat

        stats["sequence_details"][seq_id] = detail

    with open(base_dir / "sequences" / "filtered_sequences.json", "w") as f:
        json.dump(stats, f, indent=2)

    log.info("✓ Saved sequence statistics")


def save_filtered_out_sequences(
    base_dir: Path,
    all_sequences: dict[str, list],
    kept_sequences: dict[str, list],
    speed_stats: dict[str, dict] | None = None,
    filter_criteria: dict | None = None,
):
    """
    Save metadata for sequences that were filtered out (rejected).

    This is useful for debugging filter behavior and understanding why
    sequences were rejected.

    Args:
        base_dir: Output directory
        all_sequences: All discovered sequences
        kept_sequences: Sequences that passed the filter
        speed_stats: Speed statistics per sequence
        filter_criteria: Dictionary with min_speed_kmh and max_speed_kmh
    """
    rejected_sequences = {
        seq_id: frames
        for seq_id, frames in all_sequences.items()
        if seq_id not in kept_sequences
    }

    stats = {
        "total_sequences_rejected": len(rejected_sequences),
        "filter_criteria": filter_criteria or {},
        "rejection_details": {},
    }

    for seq_id, frames in rejected_sequences.items():
        if not frames:
            continue
        detail = {
            "frame_count": len(frames),
            "first_captured": frames[0].captured_at_ms,
            "last_captured": frames[-1].captured_at_ms,
            "camera_type": (
                frames[0].camera_type if frames[0].camera_type else "unknown"
            ),
        }

        # Add speed statistics if available
        if speed_stats and seq_id in speed_stats:
            speed_stat = speed_stats[seq_id]
            # Convert NamedTuple to dict if needed
            if hasattr(speed_stat, "_asdict"):
                detail["speed_statistics_kmh"] = speed_stat._asdict()
            else:
                detail["speed_statistics_kmh"] = speed_stat

            # Add rejection reason
            if filter_criteria:
                max_speed = (
                    speed_stat.max_kmh
                    if hasattr(speed_stat, "max_kmh")
                    else speed_stat.get("max_kmh", 0)
                )
                min_threshold = filter_criteria.get("min_speed_kmh", 0)
                max_threshold = filter_criteria.get("max_speed_kmh", float("inf"))

                if max_speed < min_threshold:
                    detail["rejection_reason"] = (
                        f"max_speed ({max_speed:.1f} km/h) < threshold ({min_threshold} km/h)"
                    )
                elif max_speed > max_threshold:
                    detail["rejection_reason"] = (
                        f"max_speed ({max_speed:.1f} km/h) > threshold ({max_threshold} km/h)"
                    )
                else:
                    detail["rejection_reason"] = "Unknown (possibly insufficient data)"
        else:
            detail["rejection_reason"] = "No speed data available"

        stats["rejection_details"][seq_id] = detail

    with open(base_dir / "sequences" / "filtered_out_sequences.json", "w") as f:
        json.dump(stats, f, indent=2)

    log.info(f"✓ Saved {len(rejected_sequences)} rejected sequence(s) metadata")


def main():
    parser = argparse.ArgumentParser(
        description="Download Mapillary sample data for DTM pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default bbox and download all images from kept sequences (recommended)
  python scripts/download_sample_data.py --cache-imagery
  
  # Custom bbox
  python scripts/download_sample_data.py --bbox "-48.6,-27.6,-48.59,-27.59" --cache-imagery
  
  # Limit images per sequence (faster download, less dense data)
  python scripts/download_sample_data.py --cache-imagery --images-per-sequence 10
  
  # Force refresh (ignore cache)
  python scripts/download_sample_data.py --force-refresh --cache-imagery

  # Limit total images retrieved from API
  python scripts/download_sample_data.py --max-images 2000 --cache-imagery
  
  # Custom speed filtering (default: 30-120 km/h)
  python scripts/download_sample_data.py --min-speed-kmh 25 --max-speed-kmh 100 --cache-imagery

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
        default=True,
        help="Download and cache Mapillary thumbnail images (1024px). Default: True (recommended for full pipeline testing)",
    )

    parser.add_argument(
        "--no-cache-imagery",
        dest="cache_imagery",
        action="store_false",
        help="Skip downloading imagery (metadata only)",
    )

    parser.add_argument(
        "--images-per-sequence",
        type=int,
        default=None,
        help="Maximum thumbnails to cache per sequence (default: None = all images). "
        "For full pipeline testing, downloading all images is recommended for data density.",
    )

    parser.add_argument(
        "--max-sequences",
        type=int,
        help="Limit number of sequences to download (for testing)",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        help="Maximum number of images to request from Mapillary (default: 5000)",
    )

    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh (ignore cached metadata)",
    )

    parser.add_argument(
        "--token", help="Mapillary API token (or set MAPILLARY_TOKEN env var)"
    )

    parser.add_argument(
        "--min-speed-kmh",
        type=float,
        default=30.0,
        help="Minimum max-speed threshold for car sequences (default: 30 km/h)",
    )

    parser.add_argument(
        "--max-speed-kmh",
        type=float,
        default=120.0,
        help="Maximum max-speed threshold for car sequences (default: 120 km/h)",
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
    if args.cache_imagery:
        if args.images_per_sequence is None:
            print(f"Images per sequence: ALL (full data density)")
        else:
            print(f"Images per sequence: {args.images_per_sequence}")
    print(f"Speed filter: {args.min_speed_kmh}-{args.max_speed_kmh} km/h (max-speed)")
    if args.max_images:
        print(f"Max images: {args.max_images}")
    if args.max_sequences:
        print(f"Max sequences: {args.max_sequences}")
    print("=" * 70)
    print()

    # Create directory structure
    create_directory_structure(args.output_dir)
    print()

    if args.force_refresh:
        log.info("--force-refresh flag acknowledged (no cache retained in this mode).")

    try:
        token = resolve_token(args.token)
    except RuntimeError as exc:
        log.error(str(exc))
        print("\n❌ Error: Mapillary token not found!")
        print(
            "Set MAPILLARY_TOKEN environment variable, use API_TOKEN, or supply --token"
        )
        sys.exit(1)

    print()

    # Fetch metadata using simple demo logic
    try:
        log.info("Requesting image metadata via Mapillary Graph API...")
        gdf = fetch_metadata_gdf(bbox_dict, token, limit=args.max_images)
    except Exception as exc:
        log.error("Failed to retrieve metadata: %s", exc, exc_info=True)
        sys.exit(1)

    if gdf is None or gdf.empty:
        log.warning("⚠️  No imagery found in this area.")
        print("\n" + "=" * 70)
        print("No imagery found. Try:")
        print("  1. Expanding the bounding box")
        print("  2. Checking coverage at https://www.mapillary.com/app/")
        print("  3. Using a different area")
        print("=" * 70)
        sys.exit(1)

    log.info("✓ Retrieved %d images from Mapillary", len(gdf))
    metadata_path = args.output_dir / "sequences" / "metadata.geojson"
    write_raw_metadata(gdf, metadata_path)

    sequences = gdf_to_sequences(gdf, max_sequences=args.max_sequences)
    if not sequences:
        log.warning("⚠️  No valid sequences could be assembled from metadata.")
        sys.exit(1)

    log.info("✓ Assembled %d sequences from metadata", len(sequences))

    print()

    # Filter to car-only sequences
    try:
        log.info(
            f"Filtering car-only sequences ({args.min_speed_kmh}-{args.max_speed_kmh} km/h max-speed analysis)..."
        )
        filtered_sequences, speed_stats = filter_car_sequences(
            sequences,
            min_speed_kmh=args.min_speed_kmh,
            max_speed_kmh=args.max_speed_kmh,
            return_statistics=True,
        )
        log.info(f"✓ Retained {len(filtered_sequences)} car-only sequences")

        # Display detailed speed statistics per sequence
        if speed_stats:
            print()
            print("=" * 100)
            print("Speed Statistics Per Sequence (km/h)")
            print("=" * 100)
            print(
                f"{'Seq ID':<20} {'Status':<10} {'Min':>7} {'Q1':>7} {'Median':>7} {'Q3':>7} {'Max':>7} {'Mean':>7} {'Std':>7} {'N':>5}"
            )
            print("-" * 100)

            for seq_id in sorted(sequences.keys()):
                if seq_id in speed_stats:
                    stats = speed_stats[seq_id]
                    status = "✓ KEPT" if seq_id in filtered_sequences else "✗ REJECT"
                    print(
                        f"{seq_id:<20} {status:<10} "
                        f"{stats.min_kmh:>7.1f} {stats.q1_kmh:>7.1f} {stats.median_kmh:>7.1f} "
                        f"{stats.q3_kmh:>7.1f} {stats.max_kmh:>7.1f} {stats.mean_kmh:>7.1f} "
                        f"{stats.std_kmh:>7.1f} {stats.sample_count:>5}"
                    )
                else:
                    status = "✗ NO DATA"
                    print(f"{seq_id:<20} {status:<10} {'N/A'}")

            print("=" * 100)
            print(
                f"Filter criteria: max_speed must be between {args.min_speed_kmh}-{args.max_speed_kmh} km/h"
            )
            print(f"Sequences kept: {len(filtered_sequences)}/{len(sequences)}")
            print("=" * 100)

        if not filtered_sequences:
            log.warning("⚠️  No car sequences found after filtering!")
            log.warning("This area may only have pedestrian/bike imagery")
            log.warning(
                f"Try adjusting speed thresholds (current: {args.min_speed_kmh}-{args.max_speed_kmh} km/h)"
            )
    except Exception as e:
        log.error(f"Failed to filter sequences: {e}", exc_info=True)
        filtered_sequences = sequences  # Fallback to all sequences
        speed_stats = {}

    print()

    # Save sequence statistics with speed data
    try:
        total_images = sum(len(frames) for frames in sequences.values())
        save_filtered_sequences(
            args.output_dir,
            filtered_sequences,
            len(sequences),
            total_images,
            speed_stats if speed_stats else None,
        )
    except Exception as e:
        log.warning(f"Failed to save sequence stats: {e}")

    # Save filtered-out (rejected) sequences for debugging
    try:
        filter_criteria = {
            "min_speed_kmh": args.min_speed_kmh,
            "max_speed_kmh": args.max_speed_kmh,
        }
        save_filtered_out_sequences(
            args.output_dir,
            sequences,
            filtered_sequences,
            speed_stats if speed_stats else None,
            filter_criteria,
        )
    except Exception as e:
        log.warning(f"Failed to save rejected sequences: {e}")

    # Cache imagery if requested
    cached_images = 0
    if args.cache_imagery:
        try:
            if args.images_per_sequence is None:
                log.info(
                    "Caching ALL images per sequence (recommended for full pipeline testing)..."
                )
            else:
                log.info(
                    f"Caching up to {args.images_per_sequence} images per sequence..."
                )
            target_seq_ids = (
                set(filtered_sequences.keys())
                if filtered_sequences
                else set(sequences.keys())
            )
            imagery_gdf = (
                gdf[gdf["sequence"].isin(target_seq_ids)]
                if "sequence" in gdf.columns and target_seq_ids
                else gdf
            )
            cache_stats = download_imagery_for_sequences(
                imagery_gdf, args.output_dir / "imagery", args.images_per_sequence
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
