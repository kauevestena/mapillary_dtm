#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
echo "Ready: .venv/bin/python -m slope_from_mapillary --help"
