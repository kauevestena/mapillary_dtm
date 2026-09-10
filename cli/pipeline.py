"""Compatibility entry point: all commands now use the slope pipeline.

The former GPS/camera-height DTM workflow has been retired. Its implementation
remains in Git history. See README.md for the new project-file interface.
"""
from slope_from_mapillary.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
