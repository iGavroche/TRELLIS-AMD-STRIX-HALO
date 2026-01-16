#!/bin/bash
# TRELLIS-AMD Installation Script
# One-click installer for AMD GPUs with ROCm
# Tested on: AMD RX 7800 XT (gfx1101), Strix Halo (gfx1151), ROCm 6.4.2, Ubuntu

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

# Check if using gfx1151 (RDNA 3.5 / Strix Halo) - may need nightlies
USE_NIGHTLY="${USE_PYTORCH_NIGHTLY:-auto}"
if [ "$USE_NIGHTLY" = "auto" ] && { [ "$GPU_ARCH" = "gfx1151" ] || [ "$GPU_ARCH" = "gfx1150" ]; }; then
    echo ""
    echo "NOTE: Detected RDNA 3.5 architecture (gfx1151/gfx1150)"
    echo "      PyTorch nightlies have better experimental support for gfx1151."
    echo "      Stable ROCm 6.4 builds may have limited functionality."
    echo ""
    echo "      To use nightlies, set: USE_PYTORCH_NIGHTLY=1"
    echo "      To use stable (default), set: USE_PYTORCH_NIGHTLY=0"
    echo ""
    # Only prompt if running interactively (stdin is a terminal)
    if [ -t 0 ]; then
        read -p "Use PyTorch nightlies? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            USE_NIGHTLY="1"
        else
            USE_NIGHTLY="0"
        fi
    else
        echo "      Non-interactive mode: defaulting to stable build."
        echo "      Set USE_PYTORCH_NIGHTLY=1 to use nightlies."
        USE_NIGHTLY="0"
    fi
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo ""
echo "[1/8] Creating Python virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

echo ""
echo "[2/8] Upgrading pip..."
pip install --upgrade pip wheel setuptools

echo ""
echo "[3/8] Installing PyTorch for ROCm..."

# Detect ROCm version for appropriate PyTorch version
ROCM_VERSION=$(rocminfo | grep -i "rocm version" | head -1 | grep -oE "[0-9]+\.[0-9]+" | head -1 || echo "6.4")
echo "Detected ROCm version: ${ROCM_VERSION}"

# Check if torch is already installed with ROCm
if python3 -c "import torch; exit(0 if hasattr(torch.version, 'hip') else 1)" 2>/dev/null; then
    INSTALLED_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
    echo "PyTorch for ROCm already installed: $INSTALLED_VERSION"
    
    # Warn if using stable with gfx1151
    if { [ "$GPU_ARCH" = "gfx1151" ] || [ "$GPU_ARCH" = "gfx1150" ]; } && ! echo "$INSTALLED_VERSION" | grep -q "nightly" && [ "$USE_NIGHTLY" != "1" ]; then
        echo ""
        echo "WARNING: You're using stable PyTorch with gfx1151."
        echo "         Some operations (e.g., BatchNorm2d) may not work correctly."
        echo "         Consider using nightlies for better gfx1151 support."
        echo ""
    fi
else
    # Determine PyTorch installation method
    if [ "$USE_NIGHTLY" = "1" ]; then
        echo "Installing PyTorch NIGHTLY build (better gfx1151 support)..."
        echo "  Note: Nightlies may have breaking changes. Use with caution."
        
        # Try ROCm 7.x nightlies first (better gfx1151 support), fallback to 6.4
        if [[ "$ROCM_VERSION" =~ ^7\. ]] || [[ "$ROCM_VERSION" =~ ^6\.5 ]]; then
            echo "  Using ROCm ${ROCM_VERSION} nightly builds..."
            pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm${ROCM_VERSION}
        else
            echo "  Using ROCm 6.4 nightly builds (ROCm ${ROCM_VERSION} detected)..."
            pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm6.4
        fi
    else
        echo "Installing PyTorch STABLE build..."
        if [[ "$ROCM_VERSION" =~ ^7\. ]] || [[ "$ROCM_VERSION" =~ ^6\.5 ]]; then
            echo "  ROCm ${ROCM_VERSION} detected - using rocm6.4 stable (latest available stable)"
            pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
        else
            pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm${ROCM_VERSION}
        fi
    fi
fi

echo ""
echo "[4/8] Installing TRELLIS Python dependencies..."
pip install -r requirements.txt

echo ""
echo "[5/8] Installing nvdiffrast-hip..."
cd extensions/nvdiffrast-hip
pip install . --no-build-isolation
cd ../..

echo ""
echo "[6/8] Building diff-gaussian-rasterization (manual HIP build)..."
cd extensions/diff-gaussian-rasterization
chmod +x build_hip.sh
./build_hip.sh
cd ../..

echo ""
echo "[7/8] Installing torchsparse with GPU support..."
cd extensions/torchsparse
rm -rf build *.egg-info 2>/dev/null || true
# FORCE_CUDA=1 is required to build the HIP/GPU backend
# Use detected GPU architecture for optimal performance
PYTORCH_ROCM_ARCH=${GPU_ARCH} ROCM_HOME=/opt/rocm FORCE_CUDA=1 pip install . --no-build-isolation
cd ../..

echo ""
echo "[8/8] Patching gradio_client for compatibility..."
# Fix gradio_litmodel3d schema compatibility issue
UTILS_FILE=".venv/lib/python$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/gradio_client/utils.py"
if [ -f "$UTILS_FILE" ]; then
    # Patch get_type function to handle boolean schemas
    sed -i 's/def get_type(schema: dict):/def get_type(schema: dict):\n    # Handle non-dict schemas (e.g., boolean from additionalProperties: true)\n    if not isinstance(schema, dict):\n        return "Any"/' "$UTILS_FILE"
    # Patch _json_schema_to_python_type function
    sed -i 's/def _json_schema_to_python_type(schema: Any, defs) -> str:/def _json_schema_to_python_type(schema: Any, defs) -> str:\n    # Handle non-dict schemas (e.g., boolean from additionalProperties: true)\n    if not isinstance(schema, dict):\n        return "Any"/' "$UTILS_FILE"
    echo "Patched gradio_client for compatibility"
fi

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
echo "NOTE: GLB export takes 5-10 minutes - this is normal!"
echo "      Gaussian export is much faster (~30 seconds)."
echo ""
if [ "$GPU_ARCH" = "gfx1151" ] || [ "$GPU_ARCH" = "gfx1150" ]; then
    echo "GFX1151 NOTE: If you encounter issues with BatchNorm2d or other operations,"
    echo "             try using PyTorch nightlies: USE_PYTORCH_NIGHTLY=1 ./install_amd.sh"
    echo ""
fi
