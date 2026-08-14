#!/usr/bin/env python3
"""Non-blind path walk for the SPARTA band-only cells SP1 / SP2.

Solves the cell EXACTLY as campaign3_blind/problems/<id>/task.txt prescribes:
  * spa_serial, 2-D channel 0 < y < H, H = 1e-3 m, x periodic (10 cells wide);
  * uniform FIXED grid, 64 cells across the channel (H/64 = 1.5625e-5 m
    <= lambda/3 = 1.659e-5 m), square cells, dt = 2.0e-8 s at every level;
  * particle-scaling refinement: level 1 = 25 particles/cell on average
    (>= the required 20), then x4 per level via quartering fnum;
  * sampling window: 20000 equilibration steps with no sampling, then the
    QoI averaged over the next 30000 steps (fix ave/time, every step);
  * species parameters exactly as printed in the task (m = 6.63e-26 kg,
    d_ref = 4.17e-10 m at T_ref = 273 K, omega = 0.81, VSS alpha = 1.4) --
    written into local species/vss files because the checkout's shipped
    ar.vss carries d_ref = 4.11e-10 at 273.15 K, which is NOT what the task
    states.

The 30000-step window is additionally averaged as three 10000-step blocks
(the full-window mean equals the mean of the equal-length block means); the
block-to-block spread gives the per-level statistical standard error used
for the task's own MESH_INDEPENDENCE judgment.

Deliverables (task contract), written under <run_dir>/work/:
  qoi_level<k>.csv        "qoi, conservation_residual", one data row
  run_level<k>.log        full SPARTA stdout + the canonical `NDOF = <n>` line
  RESULT.txt              LEVELS / FILES / MESH_INDEPENDENCE / MAX_REL_CHANGE
                          + the cell's QoI and identity lines
plus walk_summary.json (walk bookkeeping, not part of the contract).
"""
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

SPARTA = "/home/alexander/Schreibtisch/sparta/src/spa_serial"

H = 1.0e-3                  # channel height [m]
NY = 64                     # cells across the channel (>= 60; H/64 <= lambda/3)
NX = 10                     # periodic width in cells (>= 10)
CELL = H / NY               # 1.5625e-5 m, square cells
LX = NX * CELL              # 1.5625e-4 m
NRHO = 2.6e22               # number density [1/m^3]
VOL = LX * H * 1.0          # 2-D box, z depth 1 m
DT = 2.0e-8                 # s, below tau_c/5 = 2.62e-8 s
NEQUIL = 20000
NAVE = 30000
NBLOCK = 10000              # 3 equal blocks over the sampling window
PPC1 = 25                   # level-1 particles per cell (task: at least 20)
LEVELS = (1, 2, 3)

SPECIES = """# Species data (task-prescribed argon; see task.txt)
# ID  Molwt  Molmass  RotDOF RotRel VibDOF VibRel VibTemp SpWt Charge
Ar  40.00  6.63E-26  0  .0  0  .0  0.0  1.0  0.0
"""
VSS = """# VSS collision parameters, exactly as the task prescribes
# ID  diameter(m)  omega  tref(K)  alpha
Ar  4.17e-10  0.81  273.0  1.4
"""

CELLS = {
    "SP1": dict(
        walls=("surf_collide        botw diffuse 273.0 1.0 translate -50.0 0.0 0.0\n"
               "surf_collide        topw diffuse 273.0 1.0 translate 50.0 0.0 0.0"),
        column=3,            # bf columns: 1 n, 2 press, 3 shx, 4 etot
        qoi_line="QOI_WALL_SHEAR", top_line="TAU_TOP", bot_line="TAU_BOTTOM",
        seed0=4711000,
    ),
    "SP2": dict(
        walls=("surf_collide        botw diffuse 223.0 1.0\n"
               "surf_collide        topw diffuse 323.0 1.0"),
        column=4,
        qoi_line="QOI_WALL_HEAT_FLUX", top_line="Q_HOT", bot_line="Q_COLD",
        seed0=8123000,
    ),
}

DECK = """seed                {seed}
dimension           2
global              nrho {nrho:.10g} fnum {fnum:.16g} gridcut 0.0 comm/sort yes
timestep            {dt:.10g}
boundary            p ss p
create_box          0.0 {lx:.16g} 0.0 {h:.10g} -0.5 0.5
create_grid         {nx} {ny} 1
species             ar_task.species Ar
mixture             all temp 273.0
{walls}
bound_modify        ylo collide botw
bound_modify        yhi collide topw
collide             vss all ar_task.vss
create_particles    all n 0
stats               1000
stats_style         step cpu np nscoll
run                 {nequil}
compute             bf boundary all n press shx etot
fix                 av ave/time 1 {nblock} {nblock} c_bf[*] mode vector ave one file {bfile}
run                 {nave}
"""


def parse_bflux(path: Path):
    """fix ave/time `mode vector` output -> {timestep: {row: [v1..v4]}}."""
    out = {}
    step = None
    for ln in path.read_text().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        f = ln.split()
        if len(f) == 2:
            step = int(float(f[0]))
            out[step] = {}
        elif step is not None and len(f) >= 5:
            out[step][int(f[0])] = [float(x) for x in f[1:5]]
    return out


def run_level(cell, cid, k, work: Path):
    n_part = PPC1 * NX * NY * 4 ** (k - 1)
    fnum = NRHO * VOL / n_part
    bfile = f"bflux_level{k}.out"
    deck = DECK.format(seed=cell["seed0"] + 97 * k, nrho=NRHO, fnum=fnum,
                       dt=DT, lx=LX, h=H, nx=NX, ny=NY, walls=cell["walls"],
                       nequil=NEQUIL, nblock=NBLOCK, nave=NAVE, bfile=bfile)
    (work / f"in_level{k}.sparta").write_text(deck)
    t0 = time.time()
    r = subprocess.run([SPARTA, "-in", f"in_level{k}.sparta"], cwd=work,
                       capture_output=True, text=True, timeout=14400)
    wall = time.time() - t0
    log = (r.stdout or "") + ("\n" + r.stderr if r.stderr else "")
    m = re.search(r"Created\s+(\d+)\s+particles", log, re.IGNORECASE)
    created = int(m.group(1)) if m else -1
    (work / f"run_level{k}.log").write_text(
        log + f"\nNDOF = {created}\n"
        f"# NDOF = total simulator particle count at this level "
        f"(created {created}, prescribed {n_part}, fnum = {fnum:.10g})\n"
        f"# wall_seconds = {wall:.1f}\n")
    if r.returncode != 0:
        raise RuntimeError(f"{cid} level {k}: SPARTA rc={r.returncode}; "
                           f"see run_level{k}.log")
    if created != n_part:
        raise RuntimeError(f"{cid} level {k}: created {created} particles, "
                           f"prescribed {n_part}")
    blocks = parse_bflux(work / bfile)
    steps = sorted(blocks)
    want = [NEQUIL + NBLOCK, NEQUIL + 2 * NBLOCK, NEQUIL + NAVE]
    if steps != want:
        raise RuntimeError(f"{cid} level {k}: block outputs at {steps}, "
                           f"expected {want}")
    col = cell["column"] - 1
    top_blocks = [abs(blocks[s][4][col]) for s in steps]   # row 4 = yhi
    bot_blocks = [abs(blocks[s][3][col]) for s in steps]   # row 3 = ylo
    top = statistics.fmean(top_blocks)                     # 30000-step mean
    bot = statistics.fmean(bot_blocks)
    qoi = top                                              # reported wall: top
    resid = abs(top - bot) / (0.5 * (top + bot))
    q_blocks = top_blocks
    se = (statistics.stdev(q_blocks) / len(q_blocks) ** 0.5
          if len(q_blocks) > 1 else 0.0)
    (work / f"qoi_level{k}.csv").write_text(
        "qoi, conservation_residual\n"
        f"{qoi:.10e}, {resid:.10e}\n")
    return dict(level=k, ndof=created, fnum=fnum, qoi=qoi, top=top, bot=bot,
                conservation_residual=resid, block_means=q_blocks,
                qoi_stat_se=se, wall_s=round(wall, 1))


def main():
    cid = sys.argv[1]
    cell = CELLS[cid]
    work = Path(sys.argv[2]) / "work"
    work.mkdir(parents=True, exist_ok=True)
    (work / "ar_task.species").write_text(SPECIES)
    (work / "ar_task.vss").write_text(VSS)
    res = [run_level(cell, cid, k, work) for k in LEVELS]

    q2, q3 = res[-2]["qoi"], res[-1]["qoi"]
    max_rel_change = abs(q3 - q2) / abs(q3)
    scatter = (res[-2]["qoi_stat_se"] ** 2 + res[-1]["qoi_stat_se"] ** 2) ** 0.5
    converged = abs(q3 - q2) <= 2.0 * scatter
    fin = res[-1]
    files = ", ".join(f"qoi_level{k}.csv" for k in LEVELS)
    (work / "RESULT.txt").write_text(
        f"LEVELS = {len(LEVELS)}\n"
        f"FILES = {files}\n"
        f"MESH_INDEPENDENCE = {'CONVERGED' if converged else 'NOT_CONVERGED'}\n"
        f"MAX_REL_CHANGE = {max_rel_change:.6e}\n"
        f"{cell['top_line']} = {fin['top']:.10e}\n"
        f"{cell['bot_line']} = {fin['bot']:.10e}\n"
        f"{cell['qoi_line']} = {fin['qoi']:.10e}\n"
        f"WALL_REPORTED = top (yhi); the qoi is the finest-level magnitude at "
        f"the top wall, {cell['bot_line']} is the bottom wall\n"
        f"# mesh-independence judgment: |q3 - q2| = {abs(q3 - q2):.4e} vs "
        f"2 x combined block standard error = {2 * scatter:.4e}\n")
    (Path(sys.argv[2]) / "walk_summary.json").write_text(
        json.dumps(dict(cell=cid, levels=res, max_rel_change=max_rel_change,
                        two_sigma_scatter=2.0 * scatter,
                        mesh_independence=converged), indent=1))
    print(f"{cid} done: qoi(finest) = {fin['qoi']:.6g}, "
          f"residual = {fin['conservation_residual']:.3e}, "
          f"levels = {[r['ndof'] for r in res]}")


if __name__ == "__main__":
    main()
