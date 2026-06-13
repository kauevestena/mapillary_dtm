#!/usr/bin/env python3
"""Validate the local Mapillary sample dataset bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def validate_sample_dataset(
    dataset_dir: Path | str,
    *,
    expected_sequences: int = 30,
    expected_images: int = 3594,
    check_readable: bool = False,
) -> dict[str, Any]:
    root = Path(dataset_dir)
    filtered_path = root / "sequences" / "filtered_sequences.json"
    imagery_root = root / "imagery"
    result: dict[str, Any] = {
        "dataset_dir": str(root),
        "exists": root.exists(),
        "expected_sequences": expected_sequences,
        "expected_images": expected_images,
        "errors": [],
        "warnings": [],
    }
    if not root.exists():
        result["errors"].append("dataset directory missing")
        result["ok"] = False
        return result
    if not filtered_path.exists():
        result["errors"].append(f"missing {filtered_path}")
        result["ok"] = False
        return result

    payload = json.loads(filtered_path.read_text(encoding="utf8"))
    details = payload.get("sequence_details") or {}
    if not isinstance(details, dict):
        result["errors"].append("sequence_details must be an object")
        result["ok"] = False
        return result

    sequence_count = len(details)
    metadata_image_count = int(sum(int(item.get("frame_count", 0)) for item in details.values()))
    image_counts = {
        path.name: len(list(path.glob("*.jpg")))
        for path in sorted(imagery_root.iterdir())
        if path.is_dir()
    } if imagery_root.exists() else {}
    jpg_count = int(sum(image_counts.values()))

    mismatches = []
    for seq_id, item in details.items():
        expected = int(item.get("frame_count", 0))
        actual = int(image_counts.get(seq_id, 0))
        if expected != actual:
            mismatches.append({"sequence": seq_id, "metadata": expected, "imagery": actual})

    extra_imagery = sorted(set(image_counts) - set(details))
    qa_dirs = {
        name: _file_count(root / name)
        for name in ("reference_dtm", "ground_truth", "outputs", "qa_reports")
    }
    qa_complete = all(count > 0 for count in qa_dirs.values())
    if not qa_complete:
        result["warnings"].append("QA/reference/output folders are incomplete")

    readable_errors: list[str] = []
    if check_readable:
        readable_errors = _check_readable_images(imagery_root, limit_per_sequence=1)

    result.update(
        {
            "sequence_count": sequence_count,
            "metadata_image_count": metadata_image_count,
            "jpg_count": jpg_count,
            "mismatched_sequences": mismatches,
            "extra_imagery_sequences": extra_imagery,
            "qa_file_counts": qa_dirs,
            "qa_complete": qa_complete,
            "readable_errors": readable_errors,
        }
    )

    if sequence_count != expected_sequences:
        result["errors"].append(f"expected {expected_sequences} kept sequences, found {sequence_count}")
    if metadata_image_count != expected_images:
        result["errors"].append(f"expected {expected_images} metadata images, found {metadata_image_count}")
    if jpg_count != expected_images:
        result["errors"].append(f"expected {expected_images} JPG files, found {jpg_count}")
    if mismatches:
        result["errors"].append(f"{len(mismatches)} sequences have metadata/imagery count mismatches")
    if extra_imagery:
        result["errors"].append(f"{len(extra_imagery)} imagery sequence dirs are not in metadata")
    if readable_errors:
        result["errors"].append(f"{len(readable_errors)} sampled images could not be read")

    result["ok"] = not result["errors"]
    return result


def _file_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file())


def _check_readable_images(imagery_root: Path, *, limit_per_sequence: int) -> list[str]:
    try:
        import cv2  # type: ignore
    except Exception:
        return ["opencv unavailable for readability checks"]
    errors: list[str] = []
    for seq_dir in sorted(path for path in imagery_root.iterdir() if path.is_dir()):
        for idx, image_path in enumerate(sorted(seq_dir.glob("*.jpg"))):
            if idx >= limit_per_sequence:
                break
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                errors.append(str(image_path))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the local sample dataset bundle.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/sample_dataset"))
    parser.add_argument("--expected-sequences", type=int, default=30)
    parser.add_argument("--expected-images", type=int, default=3594)
    parser.add_argument("--check-readable", action="store_true")
    parser.add_argument("--require-qa", action="store_true")
    args = parser.parse_args()

    result = validate_sample_dataset(
        args.dataset_dir,
        expected_sequences=args.expected_sequences,
        expected_images=args.expected_images,
        check_readable=args.check_readable,
    )
    print(json.dumps(result, indent=2))
    if args.require_qa and not result.get("qa_complete"):
        return 1
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
