#!/usr/bin/env python
"""Campaign-3 blind runner.

Executes one agent per (problem, model, condition, seed) on a BLIND task and
records everything needed for cost and energy accounting. It deliberately does
NOT grade: grading happens afterwards, offline, with grade_blind.py, which is
the only component that opens a sealed key.

Safety properties:
  * the answer keys are sealed (unreadable) while this runs — the runner
    refuses to start otherwise;
  * every run gets a fresh directory and an existing ledger is never
    overwritten (a relaunch resumes and fills gaps);
  * tokens in/out, wall time and tool calls are recorded per run;
  * the live trajectory is written as it happens, so a crash or timeout cannot
    destroy the evidence the leak audit needs.

Usage:
  ../../open-fem-agent/.venv-lg/bin/python run_blind.py \
      --model 27b --conditions BARE MCP --problems B1 B2 D1 --seed 0
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

_envf = ROOT / ".env"
if _envf.exists():
    for _l in _envf.read_text().splitlines():
        if "=" in _l and not _l.lstrip().startswith("#"):
            k, v = _l.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# The agent definition (both arms) MUST come from the checkout this campaign
# lives in. The default used to be a hard-coded path to a DIFFERENT checkout
# that exists on this machine, so any launch not going through the one
# untracked driver script silently ran an uncommitted agent while the
# pre-registration attested to this tree. Self-locate, and refuse a mismatch.
REPO = Path(os.environ.get("OASIS_REPO", str(ROOT))).resolve()
if REPO != ROOT.resolve():
    sys.exit(
        f"REFUSING TO RUN — OASIS_REPO points at {REPO}, but this campaign "
        f"lives in {ROOT.resolve()}. The agent definition, the knowledge under "
        f"test and the pre-registration must all come from one checkout. "
        f"Unset OASIS_REPO or set it to {ROOT.resolve()}.")
sys.path.insert(0, str(REPO / "langgraph_eval"))
import agent as _agent                                            # noqa: E402
from agent import build_bare_agent, build_mcp_agent               # noqa: E402
from langchain_openai import ChatOpenAI                           # noqa: E402
from langchain_core.callbacks import (UsageMetadataCallbackHandler,  # noqa: E402
                                      BaseCallbackHandler)

OR_MODELS = {
    "27b": "qwen/qwen3.5-27b",
    "122b": "qwen/qwen3.5-122b-a10b",
    "397b": "qwen/qwen3.5-397b-a17b",
}

# LangGraph counts one node execution per step, so a tool-calling agent burns
# roughly two per tool call. Named because the prompt quotes it.
#
# THERE MUST BE ONE BUDGET, AND IT IS THE CLOCK. At 250 the step cap bound
# FIRST and silently: round 2 opened with NG1-BARE cut off at 124 calls having
# used 30% of its 45 minutes — with a COMPLETE submission already on disk —
# and NG2-BARE cut off at 66% while still solving. Measured cost across the
# first eight runs of that round is 6.5-29 s per call (one outlier at 102),
# so 2700 s buys roughly 90-400 calls. A 500-call ceiling therefore sits
# above what the clock can ever reach, which is the point: the agent is told
# 45 minutes, and 45 minutes is what actually stops it.
#
# In round 1 the cap almost never bound because the models quit voluntarily at
# ~37% of budget. Telling them their budget fixed that and promptly exposed
# the second, unstated limit underneath.
RECURSION_LIMIT = 1000

# EVERY PATH HERE IS CHECKED AT PREFLIGHT — see assert_environment_is_real().
#
# This block used to interpolate f"{REPO}/.venv/bin/python" for NGSolve and
# scikit-fem. When the runner was made to self-locate, REPO became this
# checkout, which has no .venv, and every agent was handed an interpreter that
# does not exist. Ten of round 2's first twenty-one runs hit it; one OASiS run
# died 8 calls in with a finished solver script it could not execute. Telling
# an agent a tool is at a path where it is not is the same defect as promising
# it source we do not serve — it reads as an instruction and burns the budget.
_NGSOLVE_SKFEM_PY = "/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python"
_ENV_PATHS = {
    "NGSolve & scikit-fem": _NGSOLVE_SKFEM_PY,
    "FEniCSx/dolfinx": "/home/alexander/miniconda3/envs/fenics/bin/python",
    "DUNE-fem": "/home/alexander/miniconda3/envs/dune-fem-env/bin/python",
    "Kratos Multiphysics (and gmsh)": _NGSOLVE_SKFEM_PY,
    "4C binary": "/home/alexander/4C/build/4C",
    "FEBio binary": "/home/alexander/FEBio/bin/febio4",
    "deal.II build tree (DEAL_II_DIR)": "/home/alexander/dealii/build",
}

ENVIRON = (
    "\nENVIRONMENT: NGSolve & scikit-fem -> "
    f"{_NGSOLVE_SKFEM_PY} ; "
    "FEniCSx/dolfinx -> /home/alexander/miniconda3/envs/fenics/bin/python ; "
    "DUNE-fem -> /home/alexander/miniconda3/envs/dune-fem-env/bin/python ; "
    f"Kratos Multiphysics (and gmsh) -> {_NGSOLVE_SKFEM_PY} ; "
    "deal.II -> build C++ with cmake using DEAL_II_DIR=/home/alexander/dealii/build "
    "(run with LD_LIBRARY_PATH=/opt/4C-dependencies/lib) ; "
    "4C binary -> /home/alexander/4C/build/4C ; "
    "FEBio binary -> /home/alexander/FEBio/bin/febio4 .\n"
)


# An interpreter that EXISTS but cannot import its backend is the same defect
# as a path that does not exist — the agent is told a tool is there, tries it,
# and spends its budget working around it. This bit twice: ofa-v2/.venv for
# NGSolve/scikit-fem (did not exist) and /usr/bin/python3 for Kratos (exists,
# is Python 3.8, and this Kratos is built for 3.12). Two coupled runs died
# believing the second one; the runs that scored found Kratos by searching the
# filesystem themselves.
_IMPORT_CHECKS = {
    "NGSolve & scikit-fem": (_NGSOLVE_SKFEM_PY, "import ngsolve, skfem"),
    "Kratos Multiphysics (and gmsh)": (_NGSOLVE_SKFEM_PY,
                                       "import KratosMultiphysics"),
    "FEniCSx/dolfinx": ("/home/alexander/miniconda3/envs/fenics/bin/python",
                        "import dolfinx"),
    "DUNE-fem": ("/home/alexander/miniconda3/envs/dune-fem-env/bin/python",
                 "import dune.fem"),
}


def assert_backends_importable() -> None:
    """Every interpreter we name must actually import what we claim it has."""
    bad = []
    for what, (interp, stmt) in _IMPORT_CHECKS.items():
        r = subprocess.run([interp, "-c", stmt], capture_output=True,
                           text=True, timeout=300)
        if r.returncode != 0:
            bad.append(f"{what}: {interp} cannot `{stmt}` "
                       f"({(r.stderr or '').strip().splitlines()[-1][:90]})")
    if bad:
        sys.exit("REFUSING TO RUN — an interpreter we advertise cannot import "
                 "its backend:\n  - " + "\n  - ".join(bad))


def assert_environment_is_real() -> None:
    """Refuse to run if anything ENVIRON names does not exist.

    An environment string is a promise to the agent. A wrong path in it does
    not fail loudly — the agent reads it, tries it, and spends its budget
    working around a tool it was told it had.
    """
    missing = [f"{what} -> {p}" for what, p in _ENV_PATHS.items()
               if not Path(p).exists()]
    if missing:
        sys.exit("REFUSING TO RUN — the ENVIRONMENT block promises paths that "
                 "do not exist:\n  - " + "\n  - ".join(missing))

_USAGE_CB = UsageMetadataCallbackHandler()


class TruncationWatch(BaseCallbackHandler):
    """Records replies the output cap cut short.

    A truncated reply is not a model failure and must never be graded as one:
    it is the harness taking the pen out of the agent's hand mid-word. Counted
    per run and surfaced in the ledger so it can be re-run rather than scored.
    """

    def __init__(self):
        self.hits = 0

    def on_llm_end(self, response, **kw):
        try:
            for gen in (response.generations or []):
                for g in gen:
                    info = getattr(g, "generation_info", None) or {}
                    if info.get("finish_reason") == "length":
                        self.hits += 1
        except Exception:
            pass


_TRUNC_CB = TruncationWatch()


def _usage_totals():
    ti = sum(int(u.get("input_tokens", 0) or 0) for u in _USAGE_CB.usage_metadata.values())
    to = sum(int(u.get("output_tokens", 0) or 0) for u in _USAGE_CB.usage_metadata.values())
    return ti, to


# Every request reserves its max output from the SAME 262144-token window the
# history lives in. Unset, the provider reserved its own default of 65536 —
# a quarter of the window, gone whether or not the model wrote a single token,
# which is what killed FC2 BARE seed3 at 196609 input tokens. The arms are not
# equally exposed: the OASiS arm accumulates faster because one knowledge call
# can return ~23k tokens, so an unnecessarily large reservation costs OASiS
# more turns than it costs bare. Measured over 128 round-3 runs, output is 839
# tokens per call on average and 2480 in the worst whole-run average, so 16384
# is roughly six times the worst case observed and frees 49152 tokens of input.
# A cap can truncate mid-sentence, so TruncationWatch below makes that loud
# rather than letting a half-written tool call be graded as the agent's answer.
_MAX_OUT = 16384


def _or_llm(size, *, temperature, seed):
    return ChatOpenAI(base_url="https://openrouter.ai/api/v1",
                      api_key=os.environ["OPENROUTER_API_KEY"],
                      model=OR_MODELS[size], temperature=temperature, seed=seed,
                      max_tokens=_MAX_OUT,
                      timeout=600, max_retries=30,
                      callbacks=[_USAGE_CB, _TRUNC_CB])


_agent._llm = _or_llm

# MATCHED CASE-INSENSITIVELY. The provider returns lowercase
# "error code: 504" while this list carried "Error code: 5", so a gateway
# failure went unrecognised and was booked as a normal run: FC2 BARE died
# after 10 tool calls to a 504 and was graded FAILED, charging OUR outage to
# the model — and to the BARE arm, which inflates the measured uplift. Same
# defect class as the evidence patterns that were lowercased on one side only.
_INFRA_ERRS = ("APIConnectionError", "Connection error", "UnicodeDecodeError",
               "InternalServerError", "empty response",
               "error code: 5", "error code: 429", "status code: 5",
               "RateLimit", "rate limit", "ReadTimeout", "ServiceUnavailable",
               "BadGateway", "Timeout error", "overloaded",
               # our own output cap, not the model's doing — see TruncationWatch
               "OutputTruncated")


# The provider refuses a request whose input exceeds its window. That is the
# agent having accumulated too much history, not an outage — a different thing
# from both a timeout and an infrastructure fault, and worth counting per arm
# because the arms are not equally exposed to it.
_CONTEXT_ERRS = ("input length", "context length", "maximum context",
                 "context_length_exceeded", "too many tokens",
                 "reduce the length")


def _is_context_exhausted(err: str) -> bool:
    if not err:
        return False
    low = err.lower()
    return any(s in low for s in _CONTEXT_ERRS)


def _is_infra(err: str) -> bool:
    """True when the run died of OUR infrastructure, not the model's work."""
    if not err:
        return False
    low = err.lower()
    return any(s.lower() in low for s in _INFRA_ERRS)


class TrajLiveLog(BaseCallbackHandler):
    """Append tool activity to disk as it happens, so a crash or timeout
    cannot destroy the trajectory the leak audit depends on."""

    def __init__(self, path):
        self._f = open(path, "a", buffering=1, encoding="utf-8", errors="replace")

    def on_tool_start(self, serialized, input_str, **kw):
        try:
            self._f.write(f"TOOL_CALL {(serialized or {}).get('name')} | "
                          f"{str(input_str)[:600]}\n")
        except Exception:
            pass

    def on_tool_end(self, output, **kw):
        try:
            self._f.write(f"TOOL_RESULT {str(output)[:600]}\n")
        except Exception:
            pass


sys.path.insert(0, str(REPO / "src"))
from blind_eval import keyvault as _kv                             # noqa: E402


def keys_are_sealed() -> bool:
    """The runner must never execute while the answer keys are readable.

    The previous implementation returned True when ``keys/`` was MISSING or
    EMPTY, so a deleted keys tree read as sealed and the campaign would have
    started with nothing to grade against.  Absence is not a seal.  The real
    check lives in the OASiS repo, under version control and under test, and is
    imported rather than restated here -- Amendment 1 declared this fixed while
    the runner still carried the broken copy and never imported the fix.
    """
    return _kv.is_sealed(_keys_dir())


def _keys_dir():
    """The ONE answer-key location, shared with grade_blind_v2.

    The custody preflight checked HERE/"keys", which does not exist in this
    checkout — the keys deliberately live outside the repository. Result: the
    preflight failed on exists=False in every configuration, including the
    correct one. Same defect class as the grader loading its evidence gate
    from another worktree: two components resolving the same thing two ways.
    OASIS_BLIND_KEYS is the single authority, exactly as in the grader.
    """
    import os as _os
    v = _os.environ.get("OASIS_BLIND_KEYS")
    return Path(v) if v else HERE / "keys"


_WROTE_RE = re.compile(r"wrote \d+ chars to (/[^\s\"']+)")


def _quarantine_stray_scratch() -> list:
    """Move aside exactly what PRIOR RUNS wrote outside their sandbox.

    Driven by the recorded transcripts, not by a filesystem sweep. The first
    version of this searched /tmp and $HOME for campaign-shaped artefacts and
    a dry run showed it would have moved 1043 directories — including this
    session's own scratch and unrelated pytest trees. Precision matters more
    than reach here: the only directories that can contaminate a run are the
    ones an earlier run actually created, and every one of those is named in
    that run's trajectory.

    Never deletes; moves into runs_quarantine/stray_scratch/ and returns what
    it moved.
    """
    dest_root = HERE / "runs_quarantine" / "stray_scratch"
    repo, here = REPO.resolve(), HERE.resolve()
    roots = {Path("/tmp").resolve(), Path.home().resolve(),
             (Path.home() / "Schreibtisch").resolve()}
    targets, moved = set(), []
    # Quarantined runs count too: their run directory was moved aside, but
    # whatever they wrote into /tmp or $HOME is still sitting there.
    trajectories = list((HERE / "runs").glob("*/work/trajectory*.txt")) + \
        list((HERE / "runs_quarantine").glob("**/work/trajectory*.txt"))
    for traj in trajectories:
        try:
            text = traj.read_text(errors="replace")
        except OSError:
            continue
        for m in _WROTE_RE.finditer(text):
            p = Path(m.group(1))
            try:
                rp = p.resolve()
            except OSError:
                continue
            if repo in rp.parents or rp == repo:
                continue                      # inside the campaign: fine
            if "claude-" in str(rp):
                continue                      # this session's own scratch
            # The directory to move is the highest ancestor still under a
            # scatter root — /tmp/coupled_heat, not /tmp/coupled_heat/level1.
            top = None
            for anc in list(rp.parents):
                if anc in roots:
                    break
                top = anc
            if top is not None and top.exists() and top.resolve() not in roots:
                targets.add(top)
    for d in sorted(targets):
        if not d.exists() or d.resolve() == here:
            continue
        dest = dest_root / d.name
        n = 1
        while dest.exists():
            n += 1
            dest = dest_root / f"{d.name}__{n}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(d), str(dest))
            moved.append(f"{d} -> {dest.relative_to(HERE)}")
        except OSError as e:
            print(f"[preflight] could not quarantine {d}: {e}")
    return moved


def preflight_or_die(problems: list) -> None:
    """Every custody control, executed, before a single paid run starts.

    One cell run with the keys readable makes every cell's blindness arguable
    afterwards, and there is no way to prove afterwards that they were sealed.
    So this is a hard gate, it runs the checks rather than asserting them, and
    it refuses on anything it cannot verify.
    """
    import subprocess
    keys = _keys_dir()
    assert_environment_is_real()
    assert_backends_importable()

    failures = []

    if not keys_are_sealed():
        failures.append(f"answer keys at {keys} are not sealed "
                        f"(exists={keys.exists()})")
    else:
        probe = _kv.verify_unreadable(keys)
        if not probe.get("sealed"):
            failures.append(f"the seal did not survive an executed read: {probe}")

    plain = [p for p in keys.rglob("*.json")
             if not p.name.endswith(".enc")] if keys.is_dir() else []
    if plain:
        failures.append(f"{len(plain)} plaintext key file(s) still on disk")

    for src in (HERE / "build_problems.py", HERE / "build_extra.py",
                HERE / "build_coupled.py"):
        if src.exists() and os.access(src, os.R_OK):
            failures.append(f"{src.name} is readable and holds hidden fields "
                            f"as literals two directories above the agent's "
                            f"workdir")

    man = REPO / "data" / "blind_key_commitment.json"
    if not man.is_file():
        failures.append(f"no key commitment at {man}: without it, nobody can "
                        f"check the solutions were fixed before the runs")

    # The exposure sweep, executed, over the whole agent-reachable tree.
    sweep = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "blind_keys.py"), "exposure",
         "--root", str(HERE)],
        capture_output=True, text=True, timeout=600)
    if sweep.returncode != 0:
        failures.append("exposure sweep found readable solution-bearing files:\n"
                        + sweep.stdout[-1500:])

    # PRIOR RUNS' LEAVINGS ARE READABLE BY THIS RUN.
    #
    # write_file is confined to the sandbox now, but bash is not, and round 1
    # scattered participants, RESULT.txt files and solution CSVs across /tmp
    # and $HOME. Two runs then READ them: C2-MCP picked up a previous
    # attempt's RESULT.txt and CSVs in /tmp/coupled_heat, and C4-MCP read
    # C6's deal.II source out of $HOME. That is one cell's work leaking into
    # another cell's measurement.
    #
    # Quarantined by EVIDENCE, never by name — a directory is moved only if
    # it actually contains campaign-shaped artefacts — and moved, never
    # deleted, so nothing is lost.
    stray = _quarantine_stray_scratch()
    if stray:
        print(f"[preflight] quarantined {len(stray)} directory(ies) of prior "
              f"agent scratch so this run cannot read them:")
        for s in stray[:12]:
            print(f"             {s}")

    # A coupled task whose intended path has never executed measures the path.
    ready = HERE / "path_readiness.json"
    if ready.is_file():
        rd = json.loads(ready.read_text()).get("path_verified", {})
        # AN ID THE FILE HAS NEVER HEARD OF IS UNREADY, NOT READY.
        #
        # This was `rd.get(p) is False`. For a problem the file does not
        # mention, rd.get returns None, and `None is False` is False — so the
        # id was treated as VERIFIED. The gate only ever caught an id
        # explicitly recorded as unwalked.
        #
        # That makes the control a no-op for exactly the case it exists for: a
        # NEW problem, whose path has by definition never run. Renaming the
        # coupled set — which a rebalanced matrix does — would have silently
        # disabled it for every cell while the preflight printed a pass. The
        # file's own opening principle is "a coupled task whose intended
        # execution path has never run measures the path, not the agent".
        #
        # Absent is now unready. Add the id with `true` once its path has
        # actually been walked, which is the only thing that should silence it.
        unready = [p for p in problems if rd.get(p) is not True]
        if unready and not os.environ.get("OASIS_SKIP_PATH_CHECK"):
            failures.append(
                f"no throwaway non-blind run has been recorded through the "
                f"intended coupling path for {unready}. A tool bug there reads "
                f"as agent failure and is charged to the arm under test. Run "
                f"one per task, discard the result, and record it in "
                f"path_readiness.json (or set OASIS_SKIP_PATH_CHECK to "
                f"override deliberately).")

    if failures:
        sys.exit("REFUSING TO RUN — custody preflight failed:\n  - "
                 + "\n  - ".join(failures))
    print(f"[preflight] custody controls verified for {len(problems)} "
          f"problem(s); keys sealed and unreadable by execution")


def per_cell_exposure_check(run_dir: Path) -> None:
    """After every cell, sweep the tree the tooling just wrote.

    Amendment 1 found seven plaintext copies of live solutions in scratch space
    AFTER the design declared custody solved -- sub-agent transcripts that had
    quoted key files, a keys snapshot, its __pycache__.  Custody has to cover
    the space tooling writes, not only the directory the design names, and a
    sweep that runs once at the start cannot see what a run creates.
    """
    import subprocess
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "blind_keys.py"), "exposure",
         "--root", str(run_dir)],
        capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        (run_dir / "EXPOSURE_ALERT.txt").write_text(r.stdout)
        print(f"  !! EXPOSURE in {run_dir.name}: a solution is readable from "
              f"the run tree; see EXPOSURE_ALERT.txt", flush=True)


def run_one(pid: str, model: str, cond: str, seed: int, timeout_s: int) -> dict:
    run_dir = HERE / "runs" / f"{pid}_{model}_{cond}_seed{seed}"
    ledger = run_dir / "ledger.json"
    if ledger.exists():
        print(f"[{pid} {model} {cond} s{seed}] SKIP: ledger exists", flush=True)
        return json.loads(ledger.read_text())

    work = run_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    task = (HERE / "problems" / pid / "task.txt").read_text(encoding="utf-8")
    # STATE THE BUDGET. Round 1: 13 of 14 coupled OASiS runs stopped
    # VOLUNTARILY at a mean of 37% of the wall budget (floor 11.8%, 24 calls,
    # zero solver runs), and 47 statements across those transcripts invoke
    # "time constraints" or invent hour estimates — a deadline the model
    # could not observe, because nothing in the prompt or any tool result
    # mentioned one. Cells that quit had converged couplings and diagnosed
    # bugs in hand. An agent that cannot see its budget guesses, and guesses
    # low. Both arms get the identical sentence.
    budget = (
        f"\nBUDGET: you have {timeout_s // 60} minutes of wall-clock for this "
        f"task. That is the only limit that will stop you — the tool-call "
        f"ceiling ({RECURSION_LIMIT // 2}) sits above what the clock can "
        f"reach. No other deadline exists. Nothing is gained by stopping "
        f"early: if you are making progress, keep working. Write "
        f"COULD_NOT_COMPLETE only when you have genuinely exhausted what you "
        f"can do, not when the task looks long.\n")
    prompt = task + ENVIRON + budget

    _USAGE_CB.usage_metadata.clear()
    _TRUNC_CB.hits = 0
    ag = (build_bare_agent if cond == "BARE" else build_mcp_agent)(
        size=model, seed=seed, workdir=work)

    live = TrajLiveLog(work / "trajectory_live.txt")
    t0, err, final, n_calls = time.time(), None, None, 0

    async def _invoke():
        return await asyncio.wait_for(
            ag.ainvoke({"messages": [("user", prompt)]},
                       config={"recursion_limit": RECURSION_LIMIT,
                               "callbacks": [live]}),
            timeout=timeout_s)

    try:
        final = asyncio.run(_invoke())
    except asyncio.TimeoutError:
        err = f"TimeoutError: exceeded {timeout_s}s"
    except Exception as e:
        err = f"{type(e).__name__}: {e}"

    # A RUN THAT HIT THE STEP CAP IS NOT A RUN THAT FINISHED. LangGraph does
    # not raise here — it returns a normal state whose last message is
    # "Sorry, need more steps to process this request." Round 1 recorded three
    # such runs with error=null, indistinguishable in the ledger from an agent
    # that chose to stop, which is exactly the confusion the budget analysis
    # then had to unpick by hand.
    if err is None and isinstance(final, dict):
        _msgs = final.get("messages", [])
        _last = getattr(_msgs[-1], "content", "") if _msgs else ""
        if isinstance(_last, str) and "need more steps" in _last:
            err = f"RecursionLimit: exhausted {RECURSION_LIMIT} graph steps"

    try:
        msgs = (final or {}).get("messages", []) if isinstance(final, dict) else []
        lines = []
        for m in msgs:
            for tc in (getattr(m, "tool_calls", None) or []):
                lines.append(f"TOOL_CALL {tc.get('name')}")
                n_calls += 1
            c = getattr(m, "content", "") or ""
            if c:
                lines.append(str(c)[:1500])
        (work / "trajectory.txt").write_text("\n".join(lines), errors="replace")
    except Exception:
        pass
    # A TIMED-OUT RUN STILL HAS A TRANSCRIPT. `final` is None on timeout, so
    # the block above wrote nothing and trajectory.txt was 0 bytes for exactly
    # the runs whose evidence matters most — every timeout of round 1. The
    # live log was on disk the whole time; promote it.
    _traj = work / "trajectory.txt"
    _live = work / "trajectory_live.txt"
    if (not _traj.exists() or _traj.stat().st_size == 0) and _live.exists():
        try:
            _traj.write_text(
                "[reconstructed from trajectory_live.txt: the run did not "
                "exit cleanly, so no final message list existed]\n"
                + _live.read_text(errors="replace"), errors="replace")
        except Exception:
            pass
    if n_calls == 0 and (work / "trajectory_live.txt").exists():
        n_calls = sum(1 for ln in (work / "trajectory_live.txt")
                      .read_text(errors="replace").splitlines()
                      if ln.startswith("TOOL_CALL "))

    tin, tout = _usage_totals()
    rec = dict(problem=pid, model=model, condition=cond, seed=seed,
               wall_s=round(time.time() - t0, 1), tool_calls=n_calls,
               tokens_in=tin, tokens_out=tout, error=err,
               truncated_replies=_TRUNC_CB.hits,
               graded=False, note="grading is offline: run grade_blind.py")
    if _TRUNC_CB.hits and not err:
        # The cap cut a reply short. Whatever the agent produced after that is
        # an artefact of our configuration, so the run is flagged for re-run
        # instead of being graded as the agent's own work. Only when the run
        # carried no other error: a timeout that also truncated is still a
        # timeout, and overwriting it here would lose the real cause.
        rec["outcome"] = "INVALID_INFRA"
        rec["error"] = (f"OutputTruncated: {_TRUNC_CB.hits} reply(ies) hit the "
                        f"{_MAX_OUT}-token output cap")
    if _is_infra(err):
        rec["outcome"] = "INVALID_INFRA"
    elif _is_context_exhausted(err):
        # NOT infrastructure, and not a timeout: the agent filled the context
        # window and the provider refused the request. That is the agent's own
        # accumulation — this harness never trims history — so the run counts
        # as a model result, but it is labelled distinctly because it is a
        # different failure from running out of clock, and because the arms
        # are not equally exposed: the OASiS arm receives much larger tool
        # responses (a single knowledge call can return ~23k tokens). Measured
        # so far: 1 occurrence in 109 runs, in the BARE arm, so no bias yet —
        # but it is worth counting per arm at every tier.
        rec["outcome"] = "CONTEXT_EXHAUSTED"
    ledger.write_text(json.dumps(rec, indent=2))
    print(f"[{pid} {model} {cond} s{seed}] done  calls={n_calls} "
          f"tok={tin}/{tout} {rec['wall_s']}s"
          + (f"  ERR {err[:60]}" if err else ""), flush=True)
    # Custody is checked per cell, not once at launch: a run creates scratch
    # that did not exist when the campaign started.
    per_cell_exposure_check(run_dir)
    if not keys_are_sealed():
        sys.exit(f"HALTING AFTER {run_dir.name}: the answer keys became "
                 f"readable during the run. Every later cell would be "
                 f"unarguably non-blind.")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(OR_MODELS))
    ap.add_argument("--conditions", nargs="+", required=True,
                    choices=["BARE", "MCP"])
    ap.add_argument("--problems", nargs="+", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--phase", required=True,
                    choices=["development", "evaluation"],
                    help="development: refine knowledge, run post-mortems, "
                         "check convergence. evaluation: the paper's numbers — "
                         "requires a freeze marker and held-out instances.")
    ap.add_argument("--timeout", type=int, default=2700,
                    help="per-run wall-clock cap in seconds")
    a = ap.parse_args()

    preflight_or_die(a.problems)

    # Which PHASE is this? Development runs against spent instances are fine and
    # expected; an EVALUATION run must not grade a problem the knowledge was
    # tuned on. Nothing in this runner used to distinguish them, so the whole
    # protection rested on somebody remembering — and a rule that depends on
    # memory is not a rule. `--phase evaluation` demands a freeze marker and
    # refuses any instance that matches a development one by id or by the
    # fingerprint of its task text, so a rename cannot launder it.
    if a.phase == "evaluation":
        import phase as _phase
        try:
            _phase.assert_evaluation_is_clean(HERE / "problems")
        except _phase.EvaluationNotCleanError as exc:
            sys.exit(f"REFUSING TO RUN AN EVALUATION: {exc}")
        overlap = sorted(set(a.problems) & set(_phase.DEVELOPMENT))
        if overlap:
            sys.exit("REFUSING TO RUN AN EVALUATION on development instances "
                     f"{overlap}: the knowledge was shaped against these, so "
                     f"grading them measures the tuning and not the tool.")
    else:
        print("[phase: development] results are for post-mortems and "
              "convergence checking, NOT for the paper's evaluation table.")

    for pid in a.problems:
        for cond in a.conditions:
            run_one(pid, a.model, cond, a.seed, a.timeout)


if __name__ == "__main__":
    main()
