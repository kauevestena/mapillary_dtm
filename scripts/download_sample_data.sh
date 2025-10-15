#!/usr/bin/env bash
#
# Download sample data for DTM from Mapillary pipeline
# Uses the bbox defined in constants.py (Florianópolis, Brazil)
#
# Usage:
#   ./scripts/download_sample_data.sh [OPTIONS]
#
# Examples:
#   # Basic download (metadata only)
#   ./scripts/download_sample_data.sh
#
#   # Download with imagery caching
#   ./scripts/download_sample_data.sh --cache-imagery --images-per-sequence 10
#
#   # Custom bbox
#   ./scripts/download_sample_data.sh --bbox "-48.6,-27.6,-48.59,-27.59"
#
# Requirements:
#   - MAPILLARY_TOKEN environment variable must be set
#   - Virtual environment must be activated or use .venv/bin/python

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

# Use virtual environment Python if available
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif [ -n "$VIRTUAL_ENV" ]; then
    PYTHON="python"
else
    PYTHON="python3"
fi

# Run the Python script using the cli module approach
exec $PYTHON -c "
import sys
import os

# Ensure we're in the right directory
os.chdir('$PROJECT_ROOT')

# Now execute the download script
exec(open('scripts/download_sample_data_impl.py').read())
" "$@"
