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

# Check if using gfx1151 (RDNA 3.5 / Strix Halo) - use AMD architecture-specific nightlies
USE_NIGHTLY="${USE_PYTORCH_NIGHTLY:-auto}"
if [ "$USE_NIGHTLY" = "auto" ] && { [ "$GPU_ARCH" = "gfx1151" ] || [ "$GPU_ARCH" = "gfx1150" ]; }; then
    echo ""
    echo "NOTE: Detected RDNA 3.5 architecture (gfx1151/gfx1150)"
    echo "      AMD provides architecture-specific PyTorch nightlies for optimal gfx1151 support."
    echo "      These will be used automatically (recommended)."
    echo ""
    echo "      To force stable builds instead, set: USE_PYTORCH_NIGHTLY=0"
    echo "      (Note: Stable builds may have limited gfx1151 functionality)"
    echo ""
    # Auto-use nightlies for gfx1151 (they're architecture-specific and recommended)
    USE_NIGHTLY="1"
elif [ "$USE_NIGHTLY" = "auto" ]; then
    # For other architectures, default to stable
    USE_NIGHTLY="0"
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo ""
echo "[1/9] Creating Python virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

echo ""
echo "[2/9] Upgrading pip..."
pip install --upgrade pip wheel setuptools

echo ""
echo "[3/9] Installing PyTorch for ROCm..."

# Detect ROCm version and installation path
ROCM_VERSION=$(rocminfo | grep -i "rocm version" | head -1 | grep -oE "[0-9]+\.[0-9]+" | head -1 || echo "6.4")
echo "Detected ROCm version: ${ROCM_VERSION}"

# Detect ROCm installation directory
if [ -d "/opt/rocm-7.1.1" ]; then
    ROCM_HOME="/opt/rocm-7.1.1"
elif [ -d "/opt/rocm" ]; then
    ROCM_HOME="/opt/rocm"
else
    # Try to find any rocm installation
    ROCM_HOME=$(ls -d /opt/rocm* 2>/dev/null | head -1)
    if [ -z "$ROCM_HOME" ]; then
        echo "WARNING: Could not find ROCm installation, defaulting to /opt/rocm"
        ROCM_HOME="/opt/rocm"
    fi
fi
echo "Detected ROCm installation: ${ROCM_HOME}"

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
    if [ "$USE_NIGHTLY" = "1" ] || [[ "$GPU_ARCH" == "gfx1151" || "$GPU_ARCH" == "gfx1150" ]]; then
        # For gfx1151/gfx1150, use AMD's architecture-specific nightlies (recommended)
        if [[ "$GPU_ARCH" == "gfx1151" || "$GPU_ARCH" == "gfx1150" ]]; then
            echo "Installing PyTorch NIGHTLY build for ${GPU_ARCH} (AMD architecture-specific builds)..."
            echo "  Using AMD ROCm nightlies optimized for ${GPU_ARCH}"
            echo "  Index: https://rocm.nightlies.amd.com/v2/${GPU_ARCH}/"
            
            # Check if uv is available (faster), otherwise use pip
            if command -v uv &> /dev/null; then
                echo "  Using uv for faster installation..."
                uv pip install --index-url "https://rocm.nightlies.amd.com/v2/${GPU_ARCH}/" --pre torch torchvision --upgrade
            else
                echo "  Using pip (consider installing 'uv' for faster installs: pip install uv)"
                pip install --pre torch torchvision --index-url "https://rocm.nightlies.amd.com/v2/${GPU_ARCH}/" --upgrade
            fi
        elif [ "$USE_NIGHTLY" = "1" ]; then
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
echo "[4/9] Installing TRELLIS Python dependencies..."
pip install -r requirements.txt

echo ""
echo "[5/9] Installing nvdiffrast-hip..."
cd extensions/nvdiffrast-hip
# Rebuild nvdiffrast to ensure it links against current ROCm version
echo "  Rebuilding nvdiffrast for current ROCm version..."
pip uninstall -y nvdiffrast 2>/dev/null || true
pip install . --no-build-isolation
cd ../..

echo ""
echo "[6/9] Building diff-gaussian-rasterization (manual HIP build)..."
cd extensions/diff-gaussian-rasterization
chmod +x build_hip.sh
./build_hip.sh
cd ../..

echo ""
echo "[7/9] Installing torchsparse with GPU support..."
cd extensions/torchsparse
rm -rf build *.egg-info 2>/dev/null || true
# FORCE_CUDA=1 is required to build the HIP/GPU backend
# Use detected GPU architecture and ROCm home for optimal performance
echo "  Building torchsparse for ${GPU_ARCH} with ROCm at ${ROCM_HOME}..."
PYTORCH_ROCM_ARCH=${GPU_ARCH} ROCM_HOME=${ROCM_HOME} FORCE_CUDA=1 pip install . --no-build-isolation
cd ../..

echo ""
echo "[8/9] Setting up library compatibility symlinks..."
# Create symlink for libamdhip64.so.6 -> .so.7 compatibility
# Some extensions were compiled against .so.6 but can use .so.7
USER_LIB_DIR="${HOME}/.local/lib"
mkdir -p "${USER_LIB_DIR}"
SYMLINK_PATH="${USER_LIB_DIR}/libamdhip64.so.6"

# Find .so.7 library
SO7_PATH=""
for lib_path in "${ROCM_HOME}/lib" "/opt/rocm/lib" "/opt/rocm-7.1.1/lib"; do
    if [ -f "${lib_path}/libamdhip64.so.7" ]; then
        SO7_PATH="${lib_path}/libamdhip64.so.7"
        break
    fi
done

if [ -n "$SO7_PATH" ] && [ ! -e "$SYMLINK_PATH" ]; then
    ln -sf "$SO7_PATH" "$SYMLINK_PATH"
    echo "  Created compatibility symlink: ${SYMLINK_PATH} -> ${SO7_PATH}"
elif [ -e "$SYMLINK_PATH" ]; then
    echo "  Compatibility symlink already exists: ${SYMLINK_PATH}"
else
    echo "  WARNING: Could not find libamdhip64.so.7 to create symlink"
fi

echo ""
echo "[9/9] Patching gradio_client for compatibility..."
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
echo "  TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 ATTN_BACKEND=sdpa XFORMERS_DISABLED=1 SPARSE_BACKEND=torchsparse python app.py"
echo ""
echo "Then open http://localhost:7860 in your browser"
echo ""
echo "NOTE: GLB export takes 5-10 minutes - this is normal!"
echo "      Gaussian export is much faster (~30 seconds)."
echo ""
if [ "$GPU_ARCH" = "gfx1151" ] || [ "$GPU_ARCH" = "gfx1150" ]; then
    echo "GFX1151 NOTE: Architecture-specific PyTorch nightlies were installed."
    echo "             If you still encounter issues, ensure ROCm is properly configured."
    echo ""
    echo "             Manual install command (if needed):"
    echo "             uv pip install --index-url https://rocm.nightlies.amd.com/v2/${GPU_ARCH}/ --pre torch torchvision --upgrade"
    echo ""
fi
