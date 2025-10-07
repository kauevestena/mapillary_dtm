"""Top-level package shim for the DTM-from-Mapillary project.

This repository keeps domain modules (api, ingest, geom, ...) at the
repository root. To allow clean package imports such as
`dtm_from_mapillary.ingest.sequence_scan`, we expose the repository root as
part of the package search path.
"""
from __future__ import annotations

from pathlib import Path

__all__: list[str] = []

# Ensure the repository root is part of the package search path so that
# `dtm_from_mapillary.<submodule>` resolves to the existing top-level
# directories without physically moving them under this package directory.
_pkg_dir = Path(__file__).resolve().parent
_repo_root = _pkg_dir.parent
if str(_repo_root) not in __path__:
    __path__.append(str(_repo_root))

# Optional: expose the repository root for consumers that need the path.
REPO_ROOT = _repo_root
