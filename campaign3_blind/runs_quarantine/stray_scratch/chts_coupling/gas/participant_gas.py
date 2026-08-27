"""SPARTA (DSMC) participant for conjugate heat transfer coupling.

This is the GAS side (Dirichlet-type participant):
- IMPORTS: wall temperature from solid at interface (x = 0.001 m)
- EXPORTS: heat flux deposited on wall to solid

Physics: rarefied argon gas in closed box, DSMC simulation
Domain: rectangle (0.001, 0.002) x (0, 0.0006) meters
BCs:
  - x = 0.001: diffuse wall with imported temperature (interface)
  - x = 0.002: diffuse wall at T = 200 K
  - y = 0, y = 0.0006: periodic

Gas properties (argon):
  - Number density n = 1.3327e22 m^-3
  - Fill temperature T = 300 K
  - VSS collision model from ar.vss file

Numerics:
  - Grid cells <= lambda/3 ≈ 3.3e-5 m (using 30x18 = 540 cells)
  - At least 25 particles per cell
  - Timestep <= 1e-7 s
  - 2000 equilibration + 4000 averaging timesteps
"""
import json
import sys
import subprocess
from pathlib import Path
import numpy as np

# Problem parameters
PARTNER    = "solid"     # partner participant name
SPARTA_BIN = "/home/alexander/Schreibtisch/sparta/src/spa_serial"
DATA_DIR   = "/home/alexander/Schreibtisch/sparta/data"

# Gas domain geometry
X0, X1     = 0.001, 0.002  # gas domain x-extent [m]
Y0, Y1     = 0.0, 0.0006   # gas domain y-extent [m]
IFACE_X    = 0.001         # interface location (left boundary of gas)

# Grid and numerics
NX, NY     = 30, 18        # grid cells (dx=3.33e-5, dy=3.33e-5, both < lambda/3)
N_PARTICLES_PER_CELL = 25  # target
DT         = 1e-7          # timestep [s]
NEQ        = 2000          # equilibration timesteps
NAVG       = 4000          # averaging timesteps
NRUN       = NEQ + NAVG    # total timesteps

# Gas properties
N_RHO      = 1.3327e22     # number density [m^-3]
T_FILL     = 300.0         # fill temperature [K]
T_RIGHT    = 200.0         # right wall temperature [K]
T_INIT     = 300.0         # iteration-1 fallback interface temperature [K]

# Derived: fnum for desired particles per cell
DX = (X1 - X0) / NX
DY = (Y1 - Y0) / NY
CELL_VOL = DX * DY * 1.0  # per meter depth in 2D
FNUM = N_PARTICLES_PER_CELL / (N_RHO * CELL_VOL)

print(f"[gas setup] Cell size: {DX:.2e} x {DY:.2e} m")
print(f"[gas setup] Target particles/cell: {N_PARTICLES_PER_CELL}, fnum: {FNUM:.2e}")


def read_imports():
    """Read imports.json; return None if not present or empty."""
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text() or "{}")
        return data.get(PARTNER) or None
    except json.JSONDecodeError:
        return None


def sample_temperature(imp, fallback, y_coords):
    """Interpolate partner's temperature values onto our interface nodes."""
    n = len(y_coords)
    if not imp or not imp.get("coordinates"):
        return np.full(n, float(fallback))
    
    ys = np.array([c[1] for c in imp["coordinates"]], float)
    ts = np.asarray(imp.get("values", []), float).ravel()
    
    if ts.size != ys.size or len(ys) < 2:
        return np.full(n, float(fallback))
    
    o = np.argsort(ys)
    return np.interp(y_coords, ys[o], ts[o])


# Create surface file for SPARTA - following exact format from circle.surf
surf_file = "walls.surf"

# Build point list
points_data = []
n_surf_points_left = NY + 1
for i in range(n_surf_points_left):
    y = Y0 + i * DY
    points_data.append((i+1, X0, y))

n_surf_points_right = NY + 1
for i in range(n_surf_points_right):
    y = Y0 + i * DY
    points_data.append((n_surf_points_left + i + 1, X1, y))

total_points = len(points_data)
total_lines = (n_surf_points_left - 1) + (n_surf_points_right - 1)

# Build line list - format is: id p1 p2 (type defaults to 1)
lines_data = []
for i in range(n_surf_points_left - 1):
    lines_data.append((i+1, i+1, i+2))

for i in range(n_surf_points_right - 1):
    p1 = n_surf_points_left + i + 1
    p2 = n_surf_points_left + i + 2
    lines_data.append((i+1, p1, p2))

# Write surface file in EXACT format matching circle.surf
with open(surf_file, 'w') as f:
    f.write("# Surface file for conjugate heat transfer\n")
    f.write("\n")
    f.write(f"{total_points} points\n")
    f.write(f"{total_lines} lines\n")
    f.write("\n")
    f.write("Points\n")
    f.write("\n")
    # Points after header
    for pid, px, py in points_data:
        f.write(f"{pid} {px:.10f} {py:.10f}\n")
    f.write("\n")
    f.write("Lines\n")
    # Lines - format: id p1 p2 (no type field)
    for lid, p1, p2 in lines_data:
        f.write(f"{lid} {p1} {p2}\n")

# Interface node coordinates (centroids of surface elements)
iface_y_centers = [Y0 + (i + 0.5) * DY for i in range(NY)]
n_iface_elements = NY

# Read imports and get interface temperature
imp = read_imports()
t_interface = sample_temperature(imp, T_INIT, iface_y_centers)

# Write custom surf temperature file for SPARTA
tsurf_file = "tsurf.in"
tsurf_lines = [
    "# Wall temperatures from coupling partner",
    "",
    f"{n_iface_elements} 1"
]
for i, t in enumerate(t_interface, 1):
    tsurf_lines.append(f"{i} {t:.10g}")

Path(tsurf_file).write_text("\n".join(tsurf_lines) + "\n")

# Copy species and VSS files
species_file = "ar.species"
vss_file = "ar.vss"
Path(species_file).write_bytes(Path(DATA_DIR, "ar.species").read_bytes())
Path(vss_file).write_bytes(Path(DATA_DIR, "ar.vss").read_bytes())

# Write SPARTA input deck
deck_file = "gas.sparta"
deck_content = f"""# SPARTA deck for conjugate heat transfer - gas side
seed                12345
dimension           2
boundary            ss p p
create_box          {X0:.10f} {X1:.10f} {Y0:.10f} {Y1:.10f} -0.5 0.5
create_grid         {NX} {NY} 1
species             {species_file} Ar
mixture             gas Ar temp {T_FILL}
global              nrho {N_RHO:.6e} fnum {FNUM:.6e}
collide             vss gas {vss_file}
timestep            {DT:.1e}

# Read surface geometry
read_surf           {surf_file} group all

# Import wall temperatures via custom surf attribute
custom              surf create tsurf float 0 file {tsurf_file} 1 tsurf

# Left wall (surface 1, interface): diffuse with imported temperature
surf_collide        1 diffuse s_tsurf 1.0

# Right wall (surface 2): diffuse at fixed temperature
surf_collide        2 diffuse {T_RIGHT} 1.0

surf_modify         all collide 1

# Create particles
create_particles    gas n 0

# Compute surface heat flux (etot = energy flux into wall)
compute             1 surf all all etot

# Average over last NAVG timesteps
fix                 1 ave/surf all 1 {NAVG} {NAVG} c_1[1] ave one

# Dump surface data every NRUN steps (just once at end)
dump                1 surf all {NRUN} flux.out id v1x v1y v2x v2y f_1 s_tsurf

# Statistics
stats               {NRUN}
stats_style         step cpu np nscoll nbound
run                 {NRUN}
"""

Path(deck_file).write_text(deck_content)

# Write NDOF (number of grid cells) to log file
n_cells = NX * NY
with open("run_level1_B.log", "w") as f:
    f.write(f"NDOF = {n_cells}\n")
    f.write(f"Grid: {NX}x{NY} = {n_cells} cells\n")
    f.write(f"Cell size: {DX:.2e} x {DY:.2e} m\n")
    f.write(f"Timestep: {DT:.1e} s\n")
    f.write(f"Total timesteps: {NRUN}\n")

print(f"[gas] Running SPARTA...")

# Run SPARTA
result = subprocess.run(
    [SPARTA_BIN, "-in", deck_file],
    capture_output=True,
    text=True,
    timeout=7200  # 2 hour timeout
)

if result.returncode != 0:
    print(f"[gas ERROR] SPARTA failed with return code {result.returncode}")
    print(f"STDOUT: {result.stdout[-2000:] if result.stdout else 'None'}")
    print(f"STDERR: {result.stderr[-2000:] if result.stderr else 'None'}")
    sys.exit(1)

# Parse the surface dump file
flux_file = "flux.out"
if not Path(flux_file).exists():
    print(f"[gas ERROR] Flux output file {flux_file} not found")
    sys.exit(1)

# Read the last snapshot from the dump
lines = Path(flux_file).read_text().splitlines()
data_start = None
for i, line in enumerate(lines):
    if line.startswith("ITEM: SURFS"):
        data_start = i + 1

if data_start is None:
    print("[gas ERROR] No SURFS block found in dump file")
    sys.exit(1)

# Parse data rows
rows = []
for line in lines[data_start:]:
    if line.startswith("ITEM:"):
        break
    parts = line.split()
    if len(parts) >= 7:
        rows.append([float(x) for x in parts[:7]])

rows = np.array(rows)
if len(rows) != n_iface_elements:
    print(f"[gas WARNING] Expected {n_iface_elements} rows, got {len(rows)}")

# Sort by element ID
rows = rows[np.argsort(rows[:, 0])]

# Extract flux and temperature
q_flux = rows[:, 5]  # etot averaged over NAVG steps
t_used = rows[:, 6]  # temperature that was actually used

mean_flux = q_flux.mean()
print(f"[gas] Flux range: [{q_flux.min():.2f}, {q_flux.max():.2f}] W/m^2")
print(f"[gas] Mean flux: {mean_flux:.2f} W/m^2")
print(f"[gas] Temperature used: [{t_used.min():.2f}, {t_used.max():.2f}] K")

# Export data
coords = [[float(IFACE_X), float(y)] for y in iface_y_centers]

Path("exports.json").write_text(json.dumps({
    "field_name": "wall_temperature",
    "n_points": int(n_iface_elements),
    "coordinates": coords,
    "values": [float(t) for t in t_used],
    "normal_fluxes": [float(q) for q in q_flux],
}, indent=2))

print(f"[gas] Exported flux to solid")
