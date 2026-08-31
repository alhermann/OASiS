"""Host hygiene: worked answers must not be readable by the agent.

Deliberately free of any LLM dependency. This lived inside run_blind.py, which
imports langchain, so the custody check could only be tested from the one
virtualenv that has langchain installed -- and the repo's suite runs from a
different one. A custody control that cannot be exercised by the normal test run
is a custody control nobody watches.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


# ── readable worked answers ───────────────────────────────────────────────
_MATERIAL_IFACE = re.compile(r"interface_level(\d+)_([AB])\.csv$")
_MATERIAL_IMPORT = re.compile(
    r"import\s+(dolfinx|ngsolve|skfem|KratosMultiphysics|dune)\b")
_MATERIAL_NATIVE = (".control", ".xplt", ".pvtu", ".vtu", ".pvd")


def _materially_useful(d: Path, files: list) -> tuple | None:
    """Is this directory a WORKED COUPLED ANSWER an agent could lift?

    Two conditions, both required, because either alone is noise:
      * a two-sided interface submission at >= 2 refinement levels -- the shape
        of a coupled deliverable set, which no unrelated artefact has;
      * AND real solver output (a native .control/.xplt/.pvtu/.vtu/.pvd) or a
        script that imports a prescribed backend -- i.e. something that
        actually ran or actually runs.

    PRECISION IS THE WHOLE POINT. A sweep on the deliverable FILENAMES alone
    matched 357 directories and 2,314 files on this machine, almost all of them
    this repo's own grader test fixtures, whose contents are synthetic data
    written to exercise the grader and teach an agent nothing. A gate that
    fires on those is a gate nobody can leave switched on. This rule matched
    exactly ONE directory on the same machine.
    """
    lv: dict = {}
    for f in files:
        m = _MATERIAL_IFACE.match(f)
        if m:
            lv.setdefault(m.group(1), set()).add(m.group(2))
    levels = sum(1 for sides in lv.values() if sides == {"A", "B"})
    if levels < 2:
        return None
    native = [f for f in files if f.endswith(_MATERIAL_NATIVE)]
    scripts = []
    for f in files:
        if f.endswith(".py"):
            try:
                if _MATERIAL_IMPORT.search((d / f).read_text(errors="ignore")):
                    scripts.append(f)
            except OSError:
                pass
    if not native and not scripts:
        return None
    return levels, len(native), scripts[:3]


def _worked_handshake(top: Path) -> tuple | None:
    """The SECOND shape: a two-code interface handshake that actually ran.

    The first shape looks for `interface_level<k>_<side>.csv` at two or more
    levels -- the campaign's own deliverable set. It is blind to the shape that
    actually leaked: three trees in /tmp,

        /tmp/fourc_kratos_cht_hlyp6gki   15 files, 0 interface_level*,
                                         4 exports/imports, 3 native 4C files

    each a COMPLETE 4C+Kratos coupled run -- slabA_4c/slabA.4C.yaml with 4C's
    own out.control and thermo VTU/PVTU, slabB_kratos/params.json, and
    exports.json plus imports.json on both sides. My gate reported "0 readable
    worked answers" while they sat there, and the same blind spot is why eleven
    copies of the identical fixture survived until they were moved by hand.

    Both conditions are required, and the second is what keeps this precise:
    2,300+ directories on this machine hold an exports/imports pair from
    driver-behaviour sweeps, nearly all of them one-point toys that teach an
    agent nothing. Requiring a NATIVE solver artefact -- a file only a real
    solver writes -- separates a run that happened from a schema demo.
    """
    # A SIZE BOUND, BECAUSE A WORKSPACE IS NOT A RUN.
    #
    # Without it this matched /home/alexander/Schreibtisch -- 14,597 native
    # artefacts -- because the walk descends through every checkout underneath
    # and trivially finds an exports/imports pair somewhere. A leaked coupled
    # run is a small self-contained directory: the leaked 4C+Kratos fixture is
    # 15 files. Anything larger than a couple of hundred is a workspace, and
    # flagging a workspace is how this sweep went wrong the first time and
    # moved a 37,445-file backend build tree.
    MAX_FILES = 200
    exports = imports = native = total = 0
    for dp, dn, fn in os.walk(top, onerror=lambda e: None):
        total += len(fn)
        if total > MAX_FILES:
            return None
        for f in fn:
            if f == "exports.json":
                exports += 1
            elif f == "imports.json":
                imports += 1
            elif f.endswith(_MATERIAL_NATIVE):
                native += 1
    if exports and imports and native:
        return exports + imports, native, []
    return None


def readable_worked_answers() -> list:
    """Worked coupled answers an agent could read, outside the campaign.

    WHY THIS EXISTS. The custody preflight refused to run while the answer KEYS
    were readable, and said nothing about worked ANSWERS lying around the host.
    Measured on this machine during development: a completed 4C+Kratos two-slab
    coupled run left by this repo's own test suite (a real 4C deck, 4C's native
    output, and exports.json/imports.json on both sides), plus seven trees of
    prior agent scatter carrying 111 files with the campaign's exact deliverable
    names. A bare agent working a coupled cell READ one of them, in a campaign
    whose headline claim is that the bare arm completes no coupled problem.

    `_quarantine_stray_scratch` cannot catch these: it moves only what a prior
    RUN's transcript names, so a developer or test artefact that no run created
    is invisible to it. This is the complementary check, and it REFUSES rather
    than moves -- deciding what to do with a developer's own files is the
    developer's call, not the runner's.
    """
    out = []
    for root in (Path("/home/alexander"), Path("/tmp"), Path("/var/tmp")):
        if not root.is_dir():
            continue
        for dp, dn, fn in os.walk(root, topdown=True, onerror=lambda e: None):
            d = Path(dp)
            s = str(d)
            if str(REPO) in s or "runs_quarantine" in s:
                dn[:] = []
                continue
            if len(d.parts) - len(root.parts) > 6:
                dn[:] = []
                continue
            got = _materially_useful(d, fn)
            if not got and (d.parent in (root, Path("/tmp"))
                            or d.parent == root):
                # tree-level check, applied at the top of each candidate tree
                got = _worked_handshake(d)
                if got:
                    dn[:] = []          # do not report every subdirectory too
            if got:
                out.append((d, got))
    return out


