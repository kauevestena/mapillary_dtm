#!/usr/bin/env bash
set -e

echo "==========================================="
echo "   mapillary_dtm Local Setup Script        "
echo "==========================================="

echo "1. Initializing git submodules..."
git submodule update --init --recursive

if [ ! -d ".venv" ]; then
    echo "2. Creating Python virtual environment (.venv)..."
    python3 -m venv .venv
else
    echo "2. Virtual environment already exists."
fi

echo "3. Activating virtual environment..."
source .venv/bin/activate

echo "4. Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "5. Installing core requirements..."
pip install -r requirements.txt

echo "6. Installing PyTorch with CUDA 12.4 support..."
# Install torch and torchvision explicitly from the CUDA 12.4 index
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "7. Installing optional ML requirements..."
pip install -r requirements-optional.txt

echo "8. Installing deep-image-matching (Track D fallback / DIM integration)..."
pip install -r requirements-dim.txt

echo "9. Validating environment..."
python scripts/check_env.py --full

echo "==========================================="
echo "Setup complete! You can now activate the environment with:"
echo "  source .venv/bin/activate"
echo "And download production models with:"
echo "  python scripts/setup_production_models.py"
echo "==========================================="
