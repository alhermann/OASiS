"""EVERY grading constant, in ONE place, each with its derivation.

This file exists because three constants in the previous grader were load-bearing
and undocumented, and one (the plausibility band) was unpacked and never read.
A constant that appears here appears NOWHERE else in the v2 grader: every module
imports it from this file, so changing a number here changes it everywhere, and
`tests/test_grader_v2_fires.py` proves the import (it monkeypatches a value here
and watches the behaviour follow).

Constants whose derivation is NOT recorded anywhere are marked
NO-RECORDED-DERIVATION rather than given an invented one. Inventing a
derivation after the fact would defeat the reason this file exists.
"""
from __future__ import annotations

# ── the grader's own evaluation set ───────────────────────────────────────
# The probe count MUST be incommensurate with every prescribed mesh level.
#
# DERIVATION (measured, recorded in DESIGN.md Amendment 2 §6): the original
# {2: 32, 3: 16} was commensurate with mesh levels 8/16/32. At the finest level
# the probe grid then has exactly the mesh's own spacing and sits at the
# midpoint of each quad's diagonal — the worst point of the P1 interpolation
# error — so the finest error is systematically over-estimated and the observed
# order biased DOWN. Measured on four manufactured fields solved on the
# prescribed sequence:
#
#     true L2 order   1.971  1.982  1.979  1.983
#     M = 32 gives    1.705  1.705  1.714  1.710      <- 0.27 of bias
#     M = 44 gives    1.969  1.982  1.980  1.983
#
# 44 and 21 share no factor with 8, 16, 32 or 64, and 44 also keeps every probe
# off x = 1/2 and x = 3/4, which 45 does not (22.5/45 = 0.5).
# `assert_probe_grid_incommensurate` in probes.py refuses a commensurate choice.
PROBE_M = {2: 44, 3: 21}

# Coordinate tolerance when matching a submitted point to a prescribed probe.
# DERIVATION: the smallest spacing between prescribed probes is
# (1/44) * (subdomain extent) ≈ 2.3e-2 for unit-order extents; the task
# prescribes at least 12 significant digits, so an honestly printed coordinate
# reproduces to ~1e-12. 1e-6 sits four orders below the spacing (no wrong-point
# match can slip in) and six orders above printing noise (no honest point is
# rejected).
PROBE_TOL = 1e-6

# ── decay quality ─────────────────────────────────────────────────────────
# PRE-REGISTERED in DESIGN.md §2 item 5: "The error sequence must be monotone
# with R² ≥ 0.98." Inherited unchanged.
MIN_R2 = 0.98

# ── absolute magnitude bounds, consulted BEFORE any rate ──────────────────
# How much larger than the exact solution's own probe-grid RMS the COARSEST
# error may be. DERIVATION (recorded in commit "blind eval: two absolute
# bounds, before the rate"): the forged case that motivated it wrote
# u + A*4**-level with errors 1e6 -> 1e3 against a solution whose RMS is 0.711
# — wrong by 1.4e6x and graded CORRECT_SUPERCONVERGENT. 5.0 is deliberately
# generous: a legitimately coarse first mesh can carry an error of the order of
# the solution itself; this bound catches orders of magnitude, not
# discretisation quality.
MAX_COARSE_REL = 5.0

# How much of the exact solution's own probe-grid RMS the FINEST error may be.
#
# NEW IN V2. The audit's finding: no bound on the finest level existed, so with
# the order at the accepted floor (theo - tol = 1.6) over 3 levels and the
# coarsest error at its own bound, the finest level could be
# 5.0 * 2**(-1.6*2) = 54% of the solution's RMS and still grade CORRECT. The
# finest level is the answer the campaign reports; an answer wrong by half of
# itself must not be CORRECT.
#
# DERIVATION (measured 2026-08-13, scratch script measure_finest.py, method:
# P1 interpolant of the sealed exact field on the finest prescribed structured
# mesh, evaluated on the grader's own probe grid — the FE error of these smooth
# problems is within a small factor of this): honest finest-level relative
# errors across live cells:
#     FE1 6.1e-4 | DL1 5.4e-4 | NG2 6.1e-3 | FC2 6.4e-3 | SK1 1.2e-2
#     C1 (worst component) 5.0e-3 | C5 (worst side) 2.2e-2
# Worst honest measurement 2.2e-2; 3-D cells at N=16 scale it by ~4x (h²), so
# the honest ceiling is ~9e-2. The bound is 0.25: >2.7x above the worst honest
# projection, and the known-bad 0.54 case fires. Like MAX_COARSE_REL it polices
# magnitude, not mesh quality.
MAX_FINEST_REL = 0.25

# ── NDOF growth under refinement ──────────────────────────────────────────
# The order estimator assumes exact mesh halving and never sees a mesh, so an
# agent that refines by 4x per "level" reads as superconvergent. The NDOF the
# run log reports must grow like 2**dim per halving. Accepted ratio band, as
# factors of 2**dim:
#
# DERIVATION: legitimate structured-mesh ratios at the prescribed levels —
#   2D P1 vertices (N+1)²: 81->289 = 3.57 (= 0.89 * 4), asymptote 4.0
#   3D P1 vertices (N+1)³: 125->729 = 5.83 (= 0.73 * 8), asymptote 8.0
#   element counts: exactly 4.0 / 8.0; P2 and mixed spaces (Taylor-Hood
#   measured 659->2467 = 3.74) approach 2**dim from below.
# Lower factor 0.55 sits under the smallest legitimate ratio (0.73 * 2**dim)
# with margin for codes that report constrained DOFs; upper factor 1.8 allows
# codes that count a mixed system generously. The forgeries this must catch:
#   unrefined (ratio 1.0)          < 0.55 * 2**dim  -> fires
#   refine-by-4 (ratio 2**(2*dim)) > 1.8  * 2**dim  -> fires (16 vs 7.2 in 2D)
NDOF_RATIO_FACTORS = (0.55, 1.8)


def ndof_ratio_bounds(dim: int) -> tuple[float, float]:
    lo, hi = NDOF_RATIO_FACTORS
    return lo * 2 ** dim, hi * 2 ** dim


# ── interface (coupled cells) ─────────────────────────────────────────────
# Worst tolerated relative two-sided interface jump (field and flux), over the
# graded levels. INHERITED from blind_eval.interface.assess's default.
# NO-RECORDED-DERIVATION as a number; the measured gap it sits in is recorded
# in DESIGN.md Amendment 2 §2: a correct partitioned coupling closes the flux
# jump to roundoff (~1e-14), every interface-mechanism mutation leaves an O(1)
# jump (0.75 to 3.0). 5e-3 separates those regimes by >11 orders on one side
# and >2 on the other; it has not been derived more finely than that.
IFACE_JUMP_TOL = 5e-3

# ── numeric guards ────────────────────────────────────────────────────────
# Below this, a reference RMS is treated as identically zero (a per-field
# magnitude check against it would divide by roundoff) and the check is
# reported NOT CHECKED for that field rather than fired or skipped silently.
ZERO_REF_FLOOR = 1e-30

# ── constants that live in the KEY, not here, and their status ────────────
# theoretical_order  — per instance, from the discretisation theory. Derived.
# tol = 0.4          — per instance in every current key. NO-RECORDED-DERIVATION
#                      (flagged by the audit; not invented here).
# band = [0.8, 3.2]  — per instance in every current key. NO-RECORDED-DERIVATION
#                      (flagged by the audit; not invented here).
# band-only / reference tolerances — pre-registered per cell inside the key
#                      (qoi.band, identity.rtol, reference.rtol); the grader
#                      refuses a grade-2/3 key that does not carry them.
