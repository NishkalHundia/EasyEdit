#!/usr/bin/env python3
"""Clear the cached axbench-concept500 dataset to fix schema mismatch issues."""

import os
import shutil

cache_dir = os.path.expanduser("~/.cache/huggingface/datasets/pyvene___axbench-concept500")

if os.path.exists(cache_dir):
    print(f"Clearing cache at: {cache_dir}")
    shutil.rmtree(cache_dir)
    print("Cache cleared successfully!")
else:
    print(f"Cache directory not found at: {cache_dir}")
    print("Nothing to clear.")

