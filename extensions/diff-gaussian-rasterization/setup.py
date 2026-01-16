#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import re
import glob
import subprocess
import torch

# Check if we're running on ROCm/HIP
is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
base_dir = os.path.dirname(os.path.abspath(__file__))

def detect_gpu_architecture():
    """Detect GPU architecture for ROCm builds.
    
    Priority:
    1. PYTORCH_ROCM_ARCH environment variable
    2. rocminfo command output
    3. Default to gfx1100 if detection fails
    
    Returns:
        str: GPU architecture identifier (e.g., 'gfx1100', 'gfx1101', 'gfx1151')
    """
    # Check environment variable first
    env_arch = os.environ.get('PYTORCH_ROCM_ARCH')
    if env_arch:
        print(f"[HIP] Using GPU architecture from PYTORCH_ROCM_ARCH: {env_arch}")
        return env_arch
    
    # Try to detect via rocminfo
    try:
        result = subprocess.run(
            ['rocminfo'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # Extract first gfx architecture found
            import re
            match = re.search(r'gfx[0-9a-z]+', result.stdout)
            if match:
                arch = match.group(0)
                print(f"[HIP] Detected GPU architecture via rocminfo: {arch}")
                return arch
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass
    
    # Default fallback
    print("[HIP] Could not detect GPU architecture, defaulting to gfx1100")
    print("[HIP] To specify manually, set PYTORCH_ROCM_ARCH environment variable")
    return "gfx1100"

def fix_hipify_issues(content):
    """Fix issues in hipified files for HIP/ROCm compatibility."""
    
    # Fix malformed kernel launch syntax: << <grid, block >> > -> hipLaunchKernelGGL
    # Actual format from hipify: '<< <grid, block >> > ('
    pattern = r'(\w+(?:<[^>]+>)?)\s*<<\s*<\s*([^,]+),\s*([^>]+)\s*>>\s*>\s*\('
    
    def convert_kernel_launch(match):
        kernel = match.group(1)
        grid = match.group(2).strip()
        block = match.group(3).strip()
        return f'hipLaunchKernelGGL({kernel}, {grid}, {block}, 0, 0, '
    
    # Apply the fix
    fixed = re.sub(pattern, convert_kernel_launch, content)
    
    # Also handle standard <<< >>> syntax
    pattern2 = r'(\w+(?:<[^>]+>)?)\s*<<<\s*([^,]+),\s*([^>]+)>>>\s*\('
    fixed = re.sub(pattern2, convert_kernel_launch, fixed)
    
    # Fix extra 0, 0 args
    fixed = re.sub(r',\s*0,\s*0,\s*0,\s*0,', ', 0, 0,', fixed)
    
    return fixed


if is_rocm:
    # For ROCm/HIP, we need to use our pre-fixed hip files as sources
    # This prevents PyTorch's CUDAExtension from re-running hipify
    hip_rasterizer_dir = os.path.join(base_dir, "hip_rasterizer")
    
    # Check if we have already hipified and fixed files
    hip_forward = os.path.join(hip_rasterizer_dir, "forward.hip")
    has_fixed_files = os.path.exists(hip_forward)
    
    if has_fixed_files:
        # Check if the files have already been fixed
        with open(hip_forward, 'r') as f:
            content = f.read()
        needs_fix = '<< <' in content  # malformed syntax indicator
        
        if needs_fix:
            print("[HIP FIX] Fixing kernel launch syntax in hipified files...")
            hip_files = glob.glob(os.path.join(hip_rasterizer_dir, "*.hip"))
            for hip_file in hip_files:
                with open(hip_file, 'r') as f:
                    content = f.read()
                fixed_content = fix_hipify_issues(content)
                if fixed_content != content:
                    print(f"[HIP FIX] Fixed: {os.path.basename(hip_file)}")
                    with open(hip_file, 'w') as f:
                        f.write(fixed_content)
        
        # Use the hipified .hip files directly as sources
        sources = [
            os.path.join(hip_rasterizer_dir, "rasterizer_impl.hip"),
            os.path.join(hip_rasterizer_dir, "forward.hip"),
            os.path.join(hip_rasterizer_dir, "backward.hip"),
            os.path.join(base_dir, "rasterize_points.hip") if os.path.exists(os.path.join(base_dir, "rasterize_points.hip")) else os.path.join(base_dir, "rasterize_points.cu"),
            os.path.join(base_dir, "ext.cpp"),
        ]
        
        # Detect GPU architecture dynamically for optimal performance
        gpu_arch = detect_gpu_architecture()
        extra_compile_args = {
            "nvcc": [
                "-I" + os.path.join(base_dir, "third_party/glm/"),
                "-I" + hip_rasterizer_dir,  # For headers
                f"--offload-arch={gpu_arch}",  # Use detected architecture (supports gfx1100, gfx1101, gfx1150, gfx1151, etc.)
                "-fgpu-rdc",  # Enable GPU relocatable device code
            ]
        }
        print(f"[HIP] Using pre-fixed hipified sources from {hip_rasterizer_dir}")
        print(f"[HIP] Compiling for architecture: {gpu_arch}")
    else:
        # First run - use original .cu files, hipify will generate them
        sources = [
            os.path.join(base_dir, "cuda_rasterizer/rasterizer_impl.cu"),
            os.path.join(base_dir, "cuda_rasterizer/forward.cu"),
            os.path.join(base_dir, "cuda_rasterizer/backward.cu"),
            os.path.join(base_dir, "rasterize_points.cu"),
            os.path.join(base_dir, "ext.cpp"),
        ]
        extra_compile_args = {
            "nvcc": [
                "-I" + os.path.join(base_dir, "third_party/glm/"),
                "-I" + os.path.join(base_dir, "cuda_rasterizer/"),
            ]
        }
        print("[HIP] First build - will use original .cu files. Run again after hipify to use fixed files.")
else:
    # CUDA path - use original files
    sources = [
        os.path.join(base_dir, "cuda_rasterizer/rasterizer_impl.cu"),
        os.path.join(base_dir, "cuda_rasterizer/forward.cu"),
        os.path.join(base_dir, "cuda_rasterizer/backward.cu"),
        os.path.join(base_dir, "rasterize_points.cu"),
        os.path.join(base_dir, "ext.cpp"),
    ]
    extra_compile_args = {
        "nvcc": ["-I" + os.path.join(base_dir, "third_party/glm/")]
    }


setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=sources,
            extra_compile_args=extra_compile_args)
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
