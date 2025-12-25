#!/bin/bash
# TRELLIS-AMD Installation Script
# One-click installer for AMD GPUs with ROCm
# Tested on: AMD RX 7800 XT, ROCm 6.4.2, Ubuntu

set -e

echo "=============================================="
echo "  TRELLIS-AMD Installation Script"
echo "  For AMD GPUs with ROCm"
echo "=============================================="

# Check for ROCm
if ! command -v rocminfo &> /dev/null; then
    echo "ERROR: ROCm not found. Please install ROCm 6.4+ first."
    echo "See: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"
    exit 1
fi

# Detect GPU
GPU_ARCH=$(rocminfo | grep -o 'gfx[0-9a-z]*' | head -1)
if [ -z "$GPU_ARCH" ]; then
    echo "WARNING: Could not detect GPU architecture, defaulting to gfx1100"
    GPU_ARCH="gfx1100"
fi
echo "Detected GPU: $GPU_ARCH"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo ""
echo "[1/7] Creating Python virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

echo ""
echo "[2/7] Upgrading pip..."
pip install --upgrade pip wheel setuptools

echo ""
echo "[3/7] Installing PyTorch for ROCm..."
# Check if torch is already installed with ROCm
if python3 -c "import torch; exit(0 if hasattr(torch.version, 'hip') else 1)" 2>/dev/null; then
    echo "PyTorch for ROCm already installed"
else
    pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
fi

echo ""
echo "[4/7] Installing TRELLIS Python dependencies..."
pip install -r requirements.txt

echo ""
echo "[5/7] Installing nvdiffrast-hip..."
cd extensions/nvdiffrast-hip
pip install . --no-build-isolation
cd ../..

echo ""
echo "[6/7] Building diff-gaussian-rasterization (manual HIP build)..."
cd extensions/diff-gaussian-rasterization
chmod +x build_hip.sh
./build_hip.sh
cd ../..

echo ""
echo "[7/7] Installing torchsparse..."
cd extensions/torchsparse
pip install . --no-build-isolation
cd ../..

echo ""
echo "=============================================="
echo "  Installation Complete!"
echo "=============================================="
echo ""
echo "To run TRELLIS:"
echo ""
echo "  source .venv/bin/activate"
echo "  ATTN_BACKEND=sdpa XFORMERS_DISABLED=1 SPARSE_BACKEND=torchsparse python app.py"
echo ""
echo "Then open http://localhost:7860 in your browser"
echo ""
