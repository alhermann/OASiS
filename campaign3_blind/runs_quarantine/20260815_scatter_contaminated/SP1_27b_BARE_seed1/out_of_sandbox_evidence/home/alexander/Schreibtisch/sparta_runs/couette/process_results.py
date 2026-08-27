#!/usr/bin/env python3
"""
Process SPARTA Couette flow simulation results and generate required output files.
"""

import os
import re
import math

def parse_surf_output(filename):
    """Parse surf_output file and extract shear stress values."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    data = {}
    in_surfs_section = False
    
    for line in lines:
        if 'ITEM: SURFS' in line:
            in_surfs_section = True
            continue
        
        if in_surfs_section and line.strip():
            parts = line.split()
            if len(parts) >= 5:
                try:
                    surf_id = int(parts[0])
                    shx = float(parts[1])  # Shear stress in x direction
                    shy = float(parts[2])
                    shz = float(parts[3])
                    press = float(parts[4])
                    data[surf_id] = {'shx': shx, 'shy': shy, 'shz': shz, 'press': press}
                except ValueError:
                    pass
    
    return data

def get_particle_count(log_filename):
    """Extract particle count from log file."""
    with open(log_filename, 'r') as f:
        content = f.read()
    
    # Look for "Created N particles" or "Particles: N ave"
    match = re.search(r'Created (\d+) particles', content)
    if match:
        return int(match.group(1))
    
    match = re.search(r'Particles:\s+(\d+)\s+ave', content)
    if match:
        return int(match.group(1))
    
    return None

def main():
    base_dir = '/home/alexander/Schreibtisch/sparta_runs/couette'
    
    levels = [1, 2, 3]
    results = {}
    
    for level in levels:
        surf_file = os.path.join(base_dir, f'surf_output_level{level}.30000')
        log_file = os.path.join(base_dir, f'log_level{level}.sparta')
        
        if not os.path.exists(surf_file):
            print(f"Error: {surf_file} not found")
            continue
        
        data = parse_surf_output(surf_file)
        if data is None or len(data) == 0:
            print(f"Error: Could not parse {surf_file}")
            continue
        
        print(f"Parsed data for level {level}: {data}")
        
        # Surface ID 1 is bottom wall, ID 2 is top wall
        tau_bottom = abs(data[1]['shx'])  # Magnitude of shear stress on bottom wall
        tau_top = abs(data[2]['shx'])     # Magnitude of shear stress on top wall
        
        # Conservation residual
        avg_tau = 0.5 * (tau_top + tau_bottom)
        if avg_tau > 0:
            conservation_residual = abs(tau_top - tau_bottom) / avg_tau
        else:
            conservation_residual = 0.0
        
        # QOI is the magnitude of shear stress (use average of both walls)
        qoi = avg_tau
        
        # Get particle count from log
        n_dof = get_particle_count(log_file)
        
        results[level] = {
            'qoi': qoi,
            'conservation_residual': conservation_residual,
            'tau_top': tau_top,
            'tau_bottom': tau_bottom,
            'n_dof': n_dof
        }
        
        print(f"Level {level}:")
        print(f"  tau_top = {tau_top:.6f} Pa")
        print(f"  tau_bottom = {tau_bottom:.6f} Pa")
        print(f"  QOI (avg) = {qoi:.6f} Pa")
        print(f"  conservation_residual = {conservation_residual:.6e}")
        print(f"  NDOF = {n_dof}")
        print()
    
    # Write CSV files for each level
    for level in levels:
        csv_file = os.path.join(base_dir, f'qoi_level{level}.csv')
        with open(csv_file, 'w') as f:
            f.write('qoi, conservation_residual\n')
            f.write(f"{results[level]['qoi']:.10e}, {results[level]['conservation_residual']:.10e}\n")
        print(f"Wrote {csv_file}")
    
    # Write log files for each level
    for level in levels:
        log_file = os.path.join(base_dir, f'run_level{level}.log')
        with open(log_file, 'w') as f:
            f.write(f"NDOF = {results[level]['n_dof']}\n")
        print(f"Wrote {log_file}")
    
    # Calculate mesh independence metrics
    # Compare finest two levels (level 2 and level 3)
    qoi_l2 = results[2]['qoi']
    qoi_l3 = results[3]['qoi']
    
    rel_change = abs(qoi_l3 - qoi_l2) / qoi_l2 if qoi_l2 > 0 else 0.0
    
    # Statistical error estimation
    # For DSMC, statistical error scales as 1/sqrt(N_particles_per_cell)
    # Level 2: ~48000 particles / 600 cells = 80 particles/cell
    # Level 3: ~192000 particles / 600 cells = 320 particles/cell
    # Statistical error at level 2: ~1/sqrt(80) ≈ 11%
    # But this is per-cell; the surface average over many collisions reduces this
    
    # More realistically, the statistical error in the time-averaged shear stress
    # depends on the number of independent samples. With 30000 timesteps and
    # many particle-wall collisions per timestep, the effective sample size is large.
    
    # Looking at the conservation residuals:
    # Level 2: 0.25% - indicates good statistical quality
    # Level 3: 0.045% - indicates excellent statistical quality
    
    # The problem states: "the qoi should move by less than the statistical scatter 
    # between the two finest levels"
    
    # A reasonable interpretation: if the relative change is comparable to or less than
    # the conservation residual at the finer level, we consider it converged.
    # At level 3, conservation_residual = 0.045%, but the actual statistical scatter
    # in the QOI itself is likely larger due to sampling variability.
    
    # Let's estimate statistical scatter from the variation between levels:
    # Level 1 -> Level 2: |1.832 - 1.843| / 1.843 = 0.6%
    # Level 2 -> Level 3: |1.851 - 1.832| / 1.832 = 1.0%
    
    # The change between level 2 and 3 is about 1%, which is consistent with
    # statistical fluctuations in DSMC. Given that the conservation residual
    # at level 3 is only 0.045%, the solution is well-resolved.
    
    # Using a practical criterion: if rel_change < 2 * conservation_residual_at_fine_level,
    # we consider it converged. This accounts for the fact that statistical scatter
    # in the QOI is typically larger than the conservation residual.
    
    stat_scatter_estimate = 2.0 * results[3]['conservation_residual']
    converged = rel_change < stat_scatter_estimate
    
    # Alternative: use a fixed threshold based on typical DSMC accuracy
    # For well-resolved DSMC, 1-2% relative change between refinement levels
    # is often considered acceptable convergence.
    converged_alt = rel_change < 0.02  # 2% threshold
    
    # Use the more conservative of the two criteria
    converged = converged and converged_alt
    
    mesh_independence = "CONVERGED" if converged else "NOT_CONVERGED"
    
    print(f"Statistical scatter estimate: {stat_scatter_estimate:.6e}")
    print(f"Relative change L2->L3: {rel_change:.6e}")
    print(f"Converged (scatter criterion): {converged}")
    print(f"Converged (2% criterion): {converged_alt}")
    
    # Write RESULT.txt
    result_file = os.path.join(base_dir, 'RESULT.txt')
    csv_files = ', '.join([f'qoi_level{k}.csv' for k in levels])
    
    with open(result_file, 'w') as f:
        f.write(f"LEVELS = {len(levels)}\n")
        f.write(f"FILES = {csv_files}\n")
        f.write(f"MESH_INDEPENDENCE = {mesh_independence}\n")
        f.write(f"MAX_REL_CHANGE = {rel_change:.10e}\n")
        f.write(f"TAU_TOP = {results[3]['tau_top']:.10e}\n")
        f.write(f"TAU_BOTTOM = {results[3]['tau_bottom']:.10e}\n")
        f.write(f"QOI_WALL_SHEAR = {results[3]['qoi']:.10e}\n")
    
    print(f"\nWrote {result_file}")
    print(f"\nSummary:")
    print(f"  MESH_INDEPENDENCE = {mesh_independence}")
    print(f"  MAX_REL_CHANGE = {rel_change:.6e}")
    print(f"  QOI at finest level = {results[3]['qoi']:.6f} Pa")

if __name__ == '__main__':
    main()
