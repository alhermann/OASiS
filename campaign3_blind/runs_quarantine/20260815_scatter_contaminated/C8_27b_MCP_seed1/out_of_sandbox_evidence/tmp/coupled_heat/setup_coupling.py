#!/usr/bin/env python3
"""Setup script for coupled simulation - prepares participant scripts and directories."""
import os
import shutil
from pathlib import Path

BASE_DIR = Path("/tmp/coupled_heat")
NGSOLVE_SCRIPT = BASE_DIR / "ngsolve_A.py"
KRATOS_SCRIPT = BASE_DIR / "kratos_B.py"

# Mesh levels
LEVELS = [8, 16, 32]

def prepare_level(level_idx, resolution):
    """Prepare directories and copy scripts for one mesh level."""
    level_dir = BASE_DIR / f"level{level_idx}"
    dir_A = level_dir / "A"
    dir_B = level_dir / "B"
    
    # Create directories
    dir_A.mkdir(parents=True, exist_ok=True)
    dir_B.mkdir(parents=True, exist_ok=True)
    
    # Copy scripts
    shutil.copy(NGSOLVE_SCRIPT, dir_A / "participant.py")
    shutil.copy(KRATOS_SCRIPT, dir_B / "participant.py")
    
    # Remove any stale imports.json files
    for d in [dir_A, dir_B]:
        imp_file = d / "imports.json"
        if imp_file.exists():
            imp_file.unlink()
    
    print(f"Prepared level {level_idx} (resolution={resolution})")
    return str(dir_A), str(dir_B)

if __name__ == "__main__":
    for i, res in enumerate(LEVELS, 1):
        prepare_level(i, res)
    print("All levels prepared!")
