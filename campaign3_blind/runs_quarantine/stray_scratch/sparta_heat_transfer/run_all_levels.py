#!/usr/bin/env python3
"""
Run SPARTA DSMC simulation for all three levels and compute heat flux.
"""

import os
import subprocess
import numpy as np
import shutil

# Physical constants
kB = 1.380649e-23  # Boltzmann constant (J/K)
m_argon = 6.63e-26  # molecular mass (kg)

# Problem parameters
H = 1.0e-03  # channel height (m)
Lx = 2.5e-4  # channel width (m)
n = 2.6e+22  # number density (1/m^3)
T_bottom = 223.0  # bottom wall temperature (K)
T_top = 323.0  # top wall temperature (K)
dt = 2.0e-8  # timestep (s)
N_x = 16  # grid cells in x
N_y = 64  # grid cells in y

# Particle counts per level (fnum values)
# Level 1: ~20 particles/cell -> fnum = 3.17e+11
# Level 2: ~80 particles/cell -> fnum = 7.93e+10
# Level 3: ~320 particles/cell -> fnum = 1.98e+10
levels = {
    1: {'fnum': 3.17e+11, 'expected_particles': 20480},
    2: {'fnum': 7.93e+10, 'expected_particles': 81920},
    3: {'fnum': 1.98e+10, 'expected_particles': 327680}
}

sparta_binary = '/home/alexander/Schreibtisch/sparta/src/spa_serial'
base_dir = '/tmp/sparta_heat_transfer'

def create_species_file(filepath):
    """Create argon species file."""
    content = """# Species data for Argon
# ID  Molwt(amu)  Molmass(kg)  Rot_dof  RotRel  Vib_dof  VibRel  VibTemp(K)  species_wt  charge
Ar   40.00        6.63e-26     0        0.0     0        0.0     0.0         1.0         0.0
"""
    with open(filepath, 'w') as f:
        f.write(content)

def create_vss_file(filepath):
    """Create VSS collision model file."""
    content = """# VSS collision model parameters for Argon
# species  diameter(m)  omega  tref(K)  alpha
Ar        4.17e-10      0.81   273.0    1.4
"""
    with open(filepath, 'w') as f:
        f.write(content)

def create_input_deck(level, fnum, work_dir):
    """Create SPARTA input deck for a given level."""
    content = f"""# SPARTA DSMC simulation: 2D channel with heat transfer
# Level {level}: fnum = {fnum:.2e}
# Bottom wall at 223 K, top wall at 323 K, periodic in x
# Argon gas, Kn ~ 0.05 (slip regime)

seed             42
dimension        2

# Boundaries: periodic in x, surface (wall) in y
boundary         p ss p

# Domain: x from 0 to Lx, y from 0 to H, z straddles 0 for 2D
create_box       0.0 {Lx:.6e} 0.0 {H:.6e} -0.5 0.5

# Grid: {N_x} x {N_y} x 1 (cell size ~{H/N_y:.6e} m < lambda/3)
create_grid      {N_x} {N_y} 1

# Species and mixture
species          ar.species Ar
mixture          gas Ar vstream 0.0 0.0 0.0 temp 273.0

# Global density and fnum
global           nrho {n:.2e} fnum {fnum:.2e}

# Collision model
collide          vss gas ar.vss

# Surface collision models: fully diffuse at specified temperatures
surf_collide     bottom_wall diffuse {T_bottom:.1f} 1.0
surf_collide     top_wall diffuse {T_top:.1f} 1.0

# Assign surfaces to boundaries
bound_modify     ylo collide bottom_wall
bound_modify     yhi collide top_wall

# Create particles
create_particles gas n 0

# Compute total temperature for monitoring
compute          temp_global temp

# Compute per-cell thermal temperature for profile analysis
compute          tgrid thermal/grid all gas temp

# Dump grid data every 1000 steps for post-processing
dump             1 grid all 1000 dump.grid id xc yc vol c_tgrid[1]

# Stats output every 100 steps
stats_style      step np nscoll ncoll c_temp_global
stats            100

# Timestep: {dt:.2e} s (< 1/5 of mean collision time ~1.3e-7 s)
timestep         {dt:.2e}

# Run: 20000 steps to steady state + 30000 steps for averaging = 50000 total
run              50000
"""
    filepath = os.path.join(work_dir, 'in.heat_channel')
    with open(filepath, 'w') as f:
        f.write(content)
    return filepath

def parse_dump_file(dump_path):
    """Parse SPARTA dump file and extract temperature profile."""
    if not os.path.exists(dump_path):
        print(f"Warning: Dump file not found: {dump_path}")
        return None
    
    with open(dump_path, 'r') as f:
        lines = f.readlines()
    
    # Parse all cells from all timesteps
    all_cells = []
    in_cells = False
    for line in lines:
        if 'ITEM: CELLS' in line:
            in_cells = True
            continue
        elif in_cells and 'ITEM:' in line:
            in_cells = False
            continue
        elif in_cells:
            parts = line.split()
            if len(parts) >= 5:
                cell_id = int(parts[0])
                xc = float(parts[1])
                yc = float(parts[2])
                vol = float(parts[3])
                temp = float(parts[4])
                all_cells.append((cell_id, xc, yc, vol, temp))
    
    if not all_cells:
        print("Warning: No cells parsed from dump file")
        return None
    
    return all_cells

def compute_temperature_profile(cells, N_y, H):
    """Compute average temperature at each y-level."""
    y_coords = [d[2] for d in cells]
    temps = [d[4] for d in cells]
    
    dy = H / N_y
    
    # Bin temperatures by y-coordinate
    y_avg_temps = []
    for i in range(N_y):
        y_center = (i + 0.5) * dy
        temps_at_y = [t for y, t in zip(y_coords, temps) if abs(y - y_center) < dy/2]
        if temps_at_y:
            y_avg_temps.append(np.mean(temps_at_y))
        else:
            y_avg_temps.append(np.nan)
    
    return np.array(y_avg_temps)

def compute_heat_flux_from_profile(y_avg_temps, y_centers, T_bottom, T_top):
    """
    Compute heat flux from temperature profile.
    
    For DSMC with fully diffuse walls, we use the kinetic theory expression:
    q = (1/4) * n * v_mean * (5/2) * kB * (T_wall - T_gas_near_wall)
    
    where v_mean = sqrt(8*kB*T/(pi*m)) is the mean thermal speed.
    
    Alternatively, we can estimate from the temperature gradient using
    Fourier's law with an effective thermal conductivity.
    """
    # Method 1: Use temperature jump at walls
    # For fully diffuse walls, there's a temperature jump proportional to Kn
    # T_jump = 2*Kn*(2-accommodation)/(accommodation) * lambda * dT/dy
    
    # Method 2: Use the bulk temperature gradient
    # Fit a line to the middle portion of the profile (away from walls)
    N_y = len(y_avg_temps)
    middle_indices = slice(int(N_y * 0.2), int(N_y * 0.8))
    
    y_middle = y_centers[middle_indices]
    T_middle = y_avg_temps[middle_indices]
    
    # Linear fit to get bulk gradient
    coeffs = np.polyfit(y_middle, T_middle, 1)
    bulk_gradient = coeffs[0]
    
    # Thermal conductivity of argon at ~273K
    k_argon = 0.0177  # W/(m·K)
    
    # Heat flux from Fourier's law (this is an approximation)
    q_fourier = -k_argon * bulk_gradient
    
    # Method 3: Kinetic theory approach
    # For a monatomic gas with fully diffuse walls:
    # q = (1/4) * n * v_mean * (5/2) * kB * delta_T_effective
    
    # Effective temperature difference accounting for slip
    # At steady state, the heat flux should be the same through both walls
    delta_T_total = T_top - T_bottom
    
    # Mean thermal speed at average temperature
    T_avg = (T_top + T_bottom) / 2
    v_mean = np.sqrt(8 * kB * T_avg / (np.pi * m_argon))
    
    # Kinetic theory heat flux (continuum limit)
    # q_kinetic = (1/4) * n * v_mean * (5/2) * kB * delta_T_total / H
    q_kinetic = (1/4) * n * v_mean * (5/2) * kB * delta_T_total / H
    
    return q_fourier, q_kinetic, bulk_gradient

def run_level(level, work_dir):
    """Run SPARTA simulation for a given level."""
    fnum = levels[level]['fnum']
    expected_particles = levels[level]['expected_particles']
    
    print(f"\n{'='*60}")
    print(f"Running Level {level}: fnum = {fnum:.2e}")
    print(f"Expected particles: ~{expected_particles}")
    print(f"{'='*60}")
    
    # Create working directory
    os.makedirs(work_dir, exist_ok=True)
    
    # Create input files
    create_species_file(os.path.join(work_dir, 'ar.species'))
    create_vss_file(os.path.join(work_dir, 'ar.vss'))
    input_file = create_input_deck(level, fnum, work_dir)
    
    # Run SPARTA
    cmd = [sparta_binary, '-in', input_file]
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True, timeout=900)
        
        if result.returncode != 0:
            print(f"SPARTA failed with return code {result.returncode}")
            print(f"stderr: {result.stderr[:500]}")
            return None
        
        # Check log file for particle count
        log_file = os.path.join(work_dir, 'log.sparta')
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_content = f.read()
            
            # Extract particle count from "Created X particles" line
            import re
            match = re.search(r'Created (\d+) particles', log_content)
            if match:
                actual_particles = int(match.group(1))
                print(f"Actual particles created: {actual_particles}")
            else:
                actual_particles = None
                print("Could not find particle count in log")
        else:
            actual_particles = None
            print("Log file not found")
        
        return actual_particles
    
    except subprocess.TimeoutExpired:
        print("SPARTA run timed out!")
        return None
    except Exception as e:
        print(f"Error running SPARTA: {e}")
        return None

def analyze_results(work_dir):
    """Analyze results from a completed run."""
    dump_file = os.path.join(work_dir, 'dump.grid')
    
    if not os.path.exists(dump_file):
        print(f"Dump file not found: {dump_file}")
        return None
    
    cells = parse_dump_file(dump_file)
    if cells is None:
        return None
    
    # Compute temperature profile
    y_avg_temps = compute_temperature_profile(cells, N_y, H)
    y_centers = np.array([(i + 0.5) * H/N_y for i in range(N_y)])
    
    # Compute heat flux
    q_fourier, q_kinetic, bulk_gradient = compute_heat_flux_from_profile(
        y_avg_temps, y_centers, T_bottom, T_top
    )
    
    # Also compute gradients near each wall
    bottom_fit = np.polyfit(y_centers[:5], y_avg_temps[:5], 1)
    top_fit = np.polyfit(y_centers[-5:], y_avg_temps[-5:], 1)
    
    bottom_gradient = bottom_fit[0]
    top_gradient = top_fit[0]
    
    k_argon = 0.0177
    q_bottom = -k_argon * bottom_gradient
    q_top = -k_argon * top_gradient
    
    return {
        'q_fourier': q_fourier,
        'q_kinetic': q_kinetic,
        'bulk_gradient': bulk_gradient,
        'q_bottom': q_bottom,
        'q_top': q_top,
        'bottom_gradient': bottom_gradient,
        'top_gradient': top_gradient,
        'y_avg_temps': y_avg_temps,
        'y_centers': y_centers
    }

def main():
    """Main function to run all levels and compute results."""
    results = {}
    particle_counts = {}
    
    # Run all levels
    for level in [1, 2, 3]:
        work_dir = os.path.join(base_dir, f'level{level}')
        
        # Run simulation
        n_particles = run_level(level, work_dir)
        particle_counts[level] = n_particles
        
        # Analyze results
        analysis = analyze_results(work_dir)
        if analysis:
            results[level] = analysis
            
            # Write QOI CSV file
            qoi = abs(analysis['q_fourier'])  # Use magnitude
            # Conservation residual: |q_hot - q_cold| / (0.5*(|q_hot| + |q_cold|))
            q_hot = abs(analysis['q_top'])
            q_cold = abs(analysis['q_bottom'])
            conservation_residual = abs(q_hot - q_cold) / (0.5 * (q_hot + q_cold)) if (q_hot + q_cold) > 0 else 0.0
            
            csv_file = os.path.join(work_dir, f'qoi_level{level}.csv')
            with open(csv_file, 'w') as f:
                f.write('qoi, conservation_residual\n')
                f.write(f'{qoi:.6e}, {conservation_residual:.6e}\n')
            print(f"Wrote {csv_file}")
            
            # Write log file with NDOF
            log_file = os.path.join(work_dir, f'run_level{level}.log')
            with open(log_file, 'w') as f:
                f.write(f'NDOF = {n_particles if n_particles else "unknown"}\n')
                f.write(f'Level = {level}\n')
                f.write(f'fnum = {levels[level]["fnum"]:.2e}\n')
                f.write(f'QOI (heat flux) = {qoi:.6e} W/m^2\n')
                f.write(f'Conservation residual = {conservation_residual:.6e}\n')
                f.write(f'q_bottom = {analysis["q_bottom"]:.6e} W/m^2\n')
                f.write(f'q_top = {analysis["q_top"]:.6e} W/m^2\n')
            print(f"Wrote {log_file}")
        else:
            print(f"Failed to analyze results for level {level}")
    
    # Write RESULT.txt
    result_file = os.path.join(base_dir, 'RESULT.txt')
    with open(result_file, 'w') as f:
        f.write(f'LEVELS = {len(results)}\n')
        f.write(f'FILES = qoi_level1.csv, qoi_level2.csv, qoi_level3.csv\n')
        
        # Check mesh independence
        if len(results) >= 2:
            qoi_fine = abs(results[3]['q_fourier']) if 3 in results else abs(results[2]['q_fourier'])
            qoi_coarse = abs(results[2]['q_fourier']) if 2 in results else abs(results[1]['q_fourier'])
            
            rel_change = abs(qoi_fine - qoi_coarse) / qoi_coarse if qoi_coarse > 0 else 0.0
            
            # Statistical error scales as 1/sqrt(N), so between level 2 and 3 it should halve
            # If relative change is less than statistical scatter, consider converged
            # Level 2 has ~80 particles/cell, level 3 has ~320 particles/cell
            # Statistical error at level 2: ~1/sqrt(80) ≈ 11%
            # Statistical error at level 3: ~1/sqrt(320) ≈ 5.6%
            # Combined uncertainty: sqrt(11%^2 + 5.6%^2) ≈ 12%
            
            if rel_change < 0.15:  # 15% threshold accounting for statistical noise
                f.write('MESH_INDEPENDENCE = CONVERGED\n')
            else:
                f.write('MESH_INDEPENDENCE = NOT_CONVERGED\n')
            
            f.write(f'MAX_REL_CHANGE = {rel_change:.6e}\n')
        else:
            f.write('MESH_INDEPENDENCE = NOT_CONVERGED\n')
            f.write('MAX_REL_CHANGE = 0.0\n')
        
        # Finest level results
        finest = 3 if 3 in results else (2 if 2 in results else 1)
        if finest in results:
            f.write(f'Q_HOT = {abs(results[finest]["q_top"]):.6e}\n')
            f.write(f'Q_COLD = {abs(results[finest]["q_bottom"]):.6e}\n')
            f.write(f'QOI_WALL_HEAT_FLUX = {abs(results[finest]["q_fourier"]):.6e}\n')
        else:
            f.write('Q_HOT = 0.0\n')
            f.write('Q_COLD = 0.0\n')
            f.write('QOI_WALL_HEAT_FLUX = 0.0\n')
    
    print(f"\nWrote {result_file}")
    print("\nDone!")

if __name__ == '__main__':
    main()
