#!/usr/bin/env python3
"""Download and record production ML model snapshots."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Iterable


MODELS = [
    {
        "name": "ground_segmentation",
        "model_id": "nvidia/segformer-b0-finetuned-cityscapes-512-1024",
        "task": "semantic-segmentation",
        "ground_labels": ["road", "sidewalk", "terrain"],
        "license": "other / NVIDIA SegFormer non-commercial research-evaluation terms",
    },
    {
        "name": "monodepth",
        "model_id": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
        "task": "depth-estimation",
        "license": "apache-2.0",
    },
]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cache production model snapshots for strict QA runs.")
    parser.add_argument(
        "--accept-model-licenses",
        action="store_true",
        help="Required acknowledgement that model licenses/terms have been reviewed and accepted.",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("models/huggingface"))
    parser.add_argument("--manifest-out", type=Path, default=Path("models/production_models.json"))
    parser.add_argument("--revision", default=None, help="Optional revision applied to both model downloads.")
    parser.add_argument("--local-files-only", action="store_true", help="Do not download; validate local cache only.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.accept_model_licenses:
        parser.error("--accept-model-licenses is required")

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise SystemExit(
            "huggingface_hub is required. Install optional dependencies with "
            "`pip install -r requirements-optional.txt`."
        ) from exc

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for item in MODELS:
        snapshot = snapshot_download(
            repo_id=item["model_id"],
            revision=args.revision,
            cache_dir=str(args.cache_dir),
            local_files_only=args.local_files_only,
        )
        snapshot_path = Path(snapshot)
        entries.append(
            {
                **item,
                "requested_revision": args.revision,
                "resolved_revision": snapshot_path.name,
                "snapshot_path": str(snapshot_path),
                "tree_sha256": _tree_sha256(snapshot_path),
            }
        )

    manifest = {
        "created_at": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
        "cache_dir": str(args.cache_dir),
        "accepted_model_licenses": True,
        "models": entries,
    }
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf8")
    print(json.dumps(manifest, indent=2))
    return 0


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode("utf8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
