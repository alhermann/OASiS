"""SPARTA path-walk participant for C13: the rarefied gas of the conjugate cell.

DIRICHLET-type role, and that is FORCED: none of SPARTA's surf_collide styles
takes a prescribed heat flux, so the gas can only RECEIVE a wall temperature
and RETURN the tallied wall energy flux. The task prescribes the roles for
exactly this reason.

THE ARRANGEMENT. A closed conduction cell: no inflow, no outflow, no stream.
The gas box's xlo boundary is the coupling interface — a diffuse wall at the
temperature imported from the solid — and xhi is a diffuse wall at T_COLD;
y is periodic. Box BOUNDARIES with `bound_modify ... collide` carry the walls,
so no embedded surface is needed and the geometry cannot leak particles.

THE FLUX TALLY. `compute boundary all etot` + `fix ave/time ... mode vector`
writes the per-boundary energy flux density (W/m^2 in SI; positive = the wall
GAINS energy from the gas). Measured in the standalone smoke test: hot wall
-1344.35, cold wall +1344.07 — an 0.02% imbalance over 4000 averaged steps,
and the magnitude sits between the slip-theory estimates with zeta = 1.875
lambda and 2.5 lambda, as it must.

SIGN. This participant exports `normal_fluxes` = etot at the interface wall:
positive = energy leaving the gas into the wall. At C13's interface the solid
is the hot side, so the export is NEGATIVE, and the solid applies it UNCHANGED
as its natural interface datum (the same apply-the-partner's-number rule every
pair here uses).

SEED. Fixed per iteration by default: the exported flux is then a
deterministic function of the imported temperature and the partitioned
iteration genuinely contracts. The sampling error is then FROZEN rather than
absent — it shifts the answer by the tally noise, which the pre-registered
band's pad covers — and the walk records the xlo/xhi imbalance as its noise
estimate.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wcommon as W                                              # noqa: E402

cfg = W.load_cfg()
if cfg["side"] != "dirichlet":
    sys.exit("SPARTA serves the Dirichlet-type role only: no surf_collide "
             "style takes a prescribed flux")
LS, LG, HH = cfg["ls"], cfg["lg"], cfg["h"]
T_COLD = cfg["t_cold"]
N_MEAN = cfg["n_mean"]
NCX, NCY = cfg.get("cells", [30, 18])
NEQUIL, NAVE = cfg.get("nequil", 2000), cfg.get("nave", 4000)
DT = cfg.get("dt", 1.0e-7)
SEED = int(cfg.get("seed", 90210))
SPARTA = cfg.get("sparta_bin",
                 "/home/alexander/Schreibtisch/sparta/src/spa_serial")
DATA = cfg.get("sparta_data", "/home/alexander/Schreibtisch/sparta/data")

for f in ("ar.species", "ar.vss"):
    if not Path(f).is_file():
        Path(f).write_bytes((Path(DATA) / f).read_bytes())

# interface points this participant reports on (the wall is 1-D in the mean;
# a short column of y-points carries the exchange)
ny = 7
ys = np.linspace(0.0, HH, ny)
ipts = np.column_stack([np.full(ny, LS), ys])

imp = W.read_imports(cfg["partner"])
Tprof = W.sample(imp, "values", ipts, cfg.get("t_init", 300.0), 1, [1]).ravel()
T_iface = float(np.mean(Tprof))

# particle weighting: aim at ~30 simulated particles per cell
vol = LG * HH                       # per metre of depth (2-D)
fnum = N_MEAN * vol / (30.0 * NCX * NCY)

deck = f"""seed                {SEED}
dimension           2
global              nrho {N_MEAN:g} fnum {fnum:.6g} gridcut 0.0 comm/sort yes
timestep            {DT:g}
boundary            ss pp p
create_box          {LS:g} {LS + LG:g} 0.0 {HH:g} -0.5 0.5
create_grid         {NCX} {NCY} 1
species             ar.species Ar
mixture             all temp 300.0

surf_collide        ifacew diffuse {T_iface:.10g} 1.0
surf_collide        coldw diffuse {T_COLD:g} 1.0
bound_modify        xlo collide ifacew
bound_modify        xhi collide coldw

collide             vss all ar.vss

create_particles    all n 0
stats               {NEQUIL}
stats_style         step cpu np nscoll

run                 {NEQUIL}

compute             bf boundary all etot
fix                 av ave/time 1 {NAVE} {NAVE + NEQUIL} c_bf[*] mode vector ave one file bflux.out
run                 {NAVE}
"""
Path("cond.sparta").write_text(deck)
r = subprocess.run([SPARTA, "-in", "cond.sparta"], capture_output=True,
                   text=True, timeout=3600)
Path("sparta_run.log").write_text((r.stdout or "")[-200000:]
                                  + "\n" + (r.stderr or "")[-20000:])
if r.returncode != 0 or not Path("bflux.out").is_file():
    sys.exit(f"SPARTA failed rc={r.returncode}; see sparta_run.log")

rows = {}
for ln in Path("bflux.out").read_text().splitlines():
    f = ln.split()
    if len(f) == 2 and not ln.startswith("#"):
        try:
            rows[int(f[0])] = float(f[1])
        except ValueError:
            pass
q_iface = rows.get(1)               # xlo = the interface wall
q_cold = rows.get(2)                # xhi = the cold wall
if q_iface is None or q_cold is None:
    sys.exit(f"bflux.out carried no xlo/xhi rows: {rows}")
imbalance = abs(q_iface + q_cold) / max(abs(q_iface), abs(q_cold), 1e-300)

W.write_log(cfg, NCX * NCY,
            f"grid_cells = {NCX * NCY}\nfnum = {fnum:.6g}\n"
            f"etot_xlo = {q_iface:.6g}\netot_xhi = {q_cold:.6g}\n"
            f"steady_imbalance_rel = {imbalance:.3e}")
print(f"[sparta {cfg['sidename']} dirichlet] cells={NCX * NCY} "
      f"T_iface={T_iface:.4f} etot_iface={q_iface:.6g} "
      f"etot_cold={q_cold:.6g} imbalance={imbalance:.3e}")
# nodes.csv: nothing field-like to measure for a DSMC side; the walk's checks
# for this cell are the band and the identity, not a nodal error.
W.write_exports(ipts, np.full((ny, 1), T_iface),
                np.full((ny, 1), q_iface), "wall_temperature")
