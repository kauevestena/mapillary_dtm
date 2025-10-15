#!/usr/bin/env python3
"""
Download Sample Data for DTM from Mapillary Pipeline

This is a launcher script that properly configures the Python path and
executes the download implementation.

Usage (from project root):
    .venv/bin/python scripts/download_sample_data.py
    .venv/bin/python scripts/download_sample_data.py --bbox "..."
    .venv/bin/python scripts/download_sample_data.py --cache-imagery
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now we can import and run the actual implementation
if __name__ == "__main__":
    # Execute the implementation module
    impl_path = Path(__file__).parent / "download_sample_data_impl.py"
    with open(impl_path) as f:
        code = compile(f.read(), impl_path, "exec")
        exec(code, {"__name__": "__main__", "__file__": str(impl_path)})
