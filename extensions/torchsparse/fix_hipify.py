#!/usr/bin/env python3
"""
Post-process hipified files to fix malformed hipLaunchKernelGGL calls.

PyTorch's hipify generates calls like:
    hipLaunchKernelGGL(kernel, grid, block, 0, 0, 0, 0, args...)
                                          ^-- extra 0, 0

The correct format is:
    hipLaunchKernelGGL(kernel, grid, block, shared_mem, stream, args...)
    hipLaunchKernelGGL(kernel, grid, block, 0, 0, args...)

This script fixes the extra arguments.
"""

import os
import re
import sys
import glob

def fix_hipify_kernel_launches(content):
    """Fix malformed hipLaunchKernelGGL calls."""
    
    # Pattern: hipLaunchKernelGGL(..., block, 0, 0, 0, 0, args)
    # Should be: hipLaunchKernelGGL(..., block, 0, 0, args) 
    # The extra "0, 0, " needs to be removed
    
    # This pattern matches ", 0, 0, 0, 0," and replaces with ", 0, 0,"
    fixed = re.sub(r',\s*0,\s*0,\s*0,\s*0,', ', 0, 0,', content)
    
    return fixed

def process_file(filepath):
    """Process a single hipified file."""
    print(f"Processing: {filepath}")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    fixed = fix_hipify_kernel_launches(content)
    
    if fixed != original:
        with open(filepath, 'w') as f:
            f.write(fixed)
        print(f"  Fixed kernel launches in {filepath}")
        return True
    else:
        print(f"  No changes needed in {filepath}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: fix_hipify.py <path_or_glob>")
        sys.exit(1)
    
    pattern = sys.argv[1]
    
    # Support glob patterns
    if '*' in pattern:
        files = glob.glob(pattern, recursive=True)
    else:
        files = [pattern]
    
    fixed_count = 0
    for filepath in files:
        if os.path.isfile(filepath):
            if process_file(filepath):
                fixed_count += 1
    
    print(f"\nFixed {fixed_count} files out of {len(files)} processed.")

if __name__ == "__main__":
    main()
