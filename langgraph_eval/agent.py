"""LangGraph agents for the open-weight ablation.

Two agent constructors:

* :func:`build_bare_agent`  — host-side toolset only (bash + web search +
  spawn_subagent). No OASiS MCP. Mirrors what Claude has in v1 BARE.
* :func:`build_mcp_agent`   — async context manager yielding the same host-side
    toolset plus the campaign OASiS MCP tools. Keep the context open for the full
    run so server-side state survives between calls.

Both conditions get parity with what Claude Code offers natively:

| Host-side tool       | Why both conditions need it                        |
|---------------------|----------------------------------------------------|
| ``run_bash``        | Equivalent of Claude's Bash; runs scripts/solvers  |
| ``read_file``       | Equivalent of Claude's Read                        |
| ``write_file``      | Equivalent of Claude's Write                       |
| ``web_search``      | Equivalent of Claude's WebSearch (literature/benchmarks) |
| ``spawn_subagent``  | Equivalent of Claude's Agent tool; needed so the   |
|                     | model can fulfil the MANDATORY CRITIC protocol the |
|                     | OASiS server prompts it to follow                  |

MCP_FULL exposes the explicit ``CAMPAIGN_MCP_TOOL_ALLOWLIST`` below. Deprecated
coupling shims and environment-mutating setup/reload tools are deliberately not
part of the experimental surface; adding a tool is a reviewed contract change,
not an automatic side effect of server registration.

All LLM calls go through ``langchain_openai.ChatOpenAI`` pointed at a local
vLLM server, which surfaces Qwen2.5's native tool-call format through the
OpenAI schema.
"""
from __future__ import annotations

import atexit
import asyncio
import hashlib
import os
import shutil
import signal
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Sequence

from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

REPO = Path(__file__).resolve().parents[1]

PORTS = {"7b": 8000, "14b": 8001, "32b": 8002}

CAMPAIGN_MCP_TOOL_ALLOWLIST = frozenset({
    "audit_results",
    "couple",
    "couple_precice",
    "developer",
    "discover",
    "examples",
    "generate_mesh",
    "knowledge",
    "prepare_simulation",
    "run_simulation",
    "run_with_generator",
    "session_insights",
    "submit_critic_review",
    "verify_mesh_independence",
    "verify_pde_consistency",
    "visualize",
})

_CRITIC_BLOCK = (
    "MANDATORY CRITIC: For every major step (problem setup, parameter "
    "choices, mesh/discretisation, BCs, solver choice, result "
    "interpretation), call `spawn_subagent` with role=\"critic\" and a "
    "ruthlessly skeptical task description. Pass the current state to the "
    "critic. Only proceed once the critic returns an explicit \"approved\" "
    "verdict. Do not approve your own work.\n"
)

BARE_SYSTEM = (
    "You are a finite-element simulation assistant. You will be given a "
    "problem statement and a writable result file path. You have host-side "
    "tools only — no FEM-aware MCP layer. Solve the problem from first "
    "principles using whatever solvers are installed on the system. Write "
    "scripts (Python / 4C YAML / etc.), run them with `run_bash`, and "
    "produce the requested RESULT lines in the result file.\n\n"
    "Tools available: run_bash, read_file, write_file, web_search, "
    "spawn_subagent.\n\n"
)
# NOTE: the mandatory-critic instruction is deliberately NOT part of the
# baseline. The critic is one of the things OASiS provides, so giving it to the
# unequipped arm hands the control group an OASiS method and understates the
# measured difference. The two arms are: host tools only (BARE) vs the OASiS
# tool layer, which includes the mandatory critic (MCP).

# THIS STRING IS THE ONLY ONE THE MODEL EVER READS.
#
# src/server.py builds an 8,919-character `instructions` block — workflow,
# mandatory critic, mesh independence, coupling. The client throws it away:
# `grep -rn "instructions"` over the installed langchain_mcp_adapters package
# returns ZERO hits, because get_tools() returns tool schemas and nothing else.
# An edit to server.py's instructions is decoration; an edit here reaches the
# agent. That was learned the hard way — audit_results was added to server.py's
# workflow as step 5 and went on being called by 1 run in 325.
#
# It also named the wrong coupling tools. `coupled_solve` and `transfer_field`
# are marked DEPRECATED in server.py and were called ZERO times in 325 MCP
# runs, while `couple` — which this prompt never mentioned — was used in 70.
MCP_SYSTEM = (
    "When you write your final RESULT.txt, it is automatically audited "
    "against your own output files and any findings appear in the write "
    "confirmation - read them, fix what is real, and rewrite the file. "
    "You can also run that check yourself at any time with "
    "`audit_results(work_dir)`: it reads ONLY your own files — no reference "
    "solution — and names the ways a complete-looking submission is wrong (a "
    "field that is identically zero because a load was never wired in, errors "
    "on a solver-tolerance floor, a convergence rate your own numbers "
    "contradict, a coupling residual history that does not actually converge). "
    "It costs one call and it works whether you ran through OASiS tools or "
    "through your own shell.\n\n"
    "You are connected to the OASiS MCP server (prepare_simulation, "
    "knowledge, discover, examples, developer, generate_mesh, run_simulation, "
    "run_with_generator, couple, couple_precice, audit_results, visualize, "
    "session_insights, submit_critic_review, verify_mesh_independence, "
    "verify_pde_consistency). "
    "Start with `prepare_simulation(solver, physics)` — it returns knowledge, "
    "real reference files and a template in one call. Use `knowledge` for "
    "physics and pitfalls (add `index=True` on the pitfalls topic: the "
    "unfiltered dump can exceed 90k tokens and will eat your context), "
    "`developer` for source lookups, and `couple` for cross-code coupling.\n\n"
    "When calling `prepare_simulation`, preserve every method-defining "
    "qualifier from the task in `physics`; do not reduce it to a generic "
    "family. For example, ask for `nearly incompressible Taylor-Hood "
    "elasticity`, `steady SIPG advection-diffusion`, or `Crank-Nicolson "
    "transient heat`, not merely `linear_elasticity`, `diffusion`, or `heat`. "
    "Those qualifiers select different spaces, operators, and templates.\n\n"
    "For every `couple` call, pass the task's absolute "
    "`residual_level<k>.csv` path as `history_path`. OASiS writes the measured "
    "finite residuals there; never retype or synthesize a history from chat "
    "output.\n\n"
    "Host-side tools (also available): run_bash, read_file, write_file, "
    "web_search, spawn_subagent.\n\n"
    + _CRITIC_BLOCK
)


# ────────────────────────────────────────────────────────────────────
# LLM factory
# ────────────────────────────────────────────────────────────────────
def _llm(size: str, *, temperature: float, seed: int) -> ChatOpenAI:
    port = PORTS[size]
    return ChatOpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="not-used-by-vllm",
        model=f"qwen2.5-{size}",
        temperature=temperature,
        seed=seed,
        timeout=600,
    )


# ────────────────────────────────────────────────────────────────────
# Host-side tools (parity with Claude Code's native surface)
# ────────────────────────────────────────────────────────────────────
# THE AGENT CANNOT SEE A CLOCK, AND IT GUESSES BADLY.
#
# The prompt states the wall-clock budget once, at the start, and nothing ever
# tells the agent how much is left. Measured on C9, 27B, seeds 22 and 23: both
# wrote COULD_NOT_COMPLETE blaming the budget — "Time constraints (45 minutes)
# prevented completion", "Insufficient time ... within the 45-minute budget" —
# after using 1093 s and 1110 s of 2700. They stopped at eighteen minutes
# believing they were out of forty-five, and threw away 59% of the run.
#
# So the harness stamps the remaining time on every command result. It is set
# by the runner, it carries no domain content, and BOTH ARMS get it: the bare
# arm builds its shell tool from this same function, and a clock is not an
# OASiS capability.
_DEADLINE = None

_SENSITIVE_ENV_MARKERS = (
    "API_KEY", "TOKEN", "SECRET", "PASSWORD", "PASSPHRASE",
    "CREDENTIAL", "AUTH", "COOKIE", "ASKPASS",
)
_OWNED_SCRATCH: set[Path] = set()


@atexit.register
def _cleanup_owned_scratch() -> None:
    for scratch in tuple(_OWNED_SCRATCH):
        shutil.rmtree(scratch, ignore_errors=True)


def _clean_subprocess_env() -> dict[str, str]:
    """Return runtime environment without credentials or key-vault paths."""
    clean = {
        key: value for key, value in os.environ.items()
        if not any(marker in key.upper() for marker in _SENSITIVE_ENV_MARKERS)
    }
    clean.pop("OASIS_BLIND_KEYS", None)
    clean.pop("SSH_AUTH_SOCK", None)
    return clean


def _pin_backend_runtime_env(env: dict[str, str], *,
                             home: Path | None = None,
                             workspace: Path | None = None) -> None:
    """Expose mounted solver runtimes after the sandbox replaces ``HOME``."""
    host_home = (home or Path.home()).resolve()
    host_workspace = (workspace or REPO.parent).resolve()
    defaults = (
        ("FENICS_PYTHON",
         host_home / "miniconda3/envs/fenics/bin/python", "file"),
        ("DUNE_PYTHON",
         host_home / "miniconda3/envs/dune-py313/bin/python", "file"),
        ("FEBIO_BINARY", host_home / "FEBio/bin/febio4", "file"),
        ("DEAL_II_DIR", host_home / "dealii/build", "dir"),
        ("SPARTA_BINARY",
         host_workspace / "sparta/src/spa_serial", "file"),
        ("SPARTA_ROOT", host_workspace / "sparta", "dir"),
        ("SPARTA_DATA_DIR", host_workspace / "sparta/data", "dir"),
    )
    for key, path, kind in defaults:
        exists = path.is_file() if kind == "file" else path.is_dir()
        if not env.get(key) and exists:
            env[key] = str(path)


_ACTION_BUDGET = None        # (limit, counter) set by the runner
_ACTIONS_USED = 0


def note_action() -> None:
    """Count one tool call, so the agent can be told what it has left."""
    global _ACTIONS_USED
    _ACTIONS_USED += 1


def _time_left_note() -> str:
    """What the agent has left -- in ACTIONS as well as minutes.

    THE CLOCK WAS NEVER THE BINDING CONSTRAINT. Measured from file mtimes over
    six coupled runs: five wrote their submission at 93-99% of their whole
    file-activity span, with 8 to 115 seconds of activity after it, and the
    only run that reached a gradeable order with both prescribed codes proven
    submitted at 68%. They run out of ACTIONS, and a note in minutes cannot
    tell a model that it has eight tool calls left of fifty.

    BATS (arXiv 2511.17006) measured the bare budget tracker at 40.4% fewer
    searches and 31.3% lower cost at a budget of ten, with ReAct going
    12.6% -> 24.6% on BrowseComp once budget awareness was added. This is the
    cheapest of the four interventions that measurement pointed at.

    Both halves are omitted when unset, so nothing is invented: an unset
    deadline or budget prints nothing rather than a guess.
    """
    parts = []
    if _DEADLINE is not None:
        import time as _t
        left = _DEADLINE[0] - _t.time()
        total = _DEADLINE[1]
        parts.append("clock: budget spent" if left <= 0 else
                     f"clock: {int(left // 60)} min left of {int(total // 60)}")
    # NO INVENTED ACTION LIMIT. The graph ceiling is RECURSION_LIMIT = 1000 and
    # run_blind.py's own comment says it "sits above what the clock can" reach,
    # so there is no action budget to count down to. Reporting "465 of 500
    # left" would tell the agent it has room while the clock is what ends the
    # run. What is true and useful is how many actions it has SPENT, and how
    # much clock remains -- so the urgency trigger is keyed on the clock.
    if _ACTIONS_USED:
        parts.append(f"actions spent: {_ACTIONS_USED}")
    if _DEADLINE is not None:
        import time as _t
        frac = (_DEADLINE[0] - _t.time()) / max(_DEADLINE[1], 1.0)
        if frac <= 0.25:
            parts.append(
                "WRITE THE DELIVERABLE NOW, with whatever levels you have. "
                "Measured over six coupled runs: five wrote their submission "
                "at 93-99% of their activity span and had 8 to 115 seconds "
                "left to fix anything; the one that reached a gradeable order "
                "with both codes proven wrote it at 68%. A partial submission "
                "is graded. An unwritten one is not.")
    return ("\n[" + " | ".join(parts) + "]") if parts else ""


def _format_audit_reply(findings) -> str:
    """The audit's wording, in ONE place, for every route that submits.

    Extracted when the shell route was added: the same submission must get the
    same message whether it was written with write_file or a heredoc, and two
    copies of this wording would drift the moment either is edited.
    """
    if findings == "NOEVIDENCE":
        return ("\n[auto-audit: found NO per-level result files "
                "to check. This is NOT a clean bill — it means "
                "there is nothing here to verify. If you have "
                "results, write them per level; if you do not, "
                "this submission has no numbers behind it.]")
    if findings:
        return ("\n\nAUTO-AUDIT of your submission (from your own "
                "files only — no reference solution):\n" + findings +
                "\nA finding is a pointer, not a verdict: check the "
                "named place, fix if real, and REWRITE this file. "
                "Submitting with a standing finding usually grades "
                "wrong.")
    if findings == "":
        return ("\n[auto-audit: clean — self-consistent, which "
                "is necessary but not sufficient for correct]")
    return ""


def _add_mount_dirs(argv: list[str], root: Path, target: Path,
                    made: set[Path]) -> None:
    """Create bubblewrap mount points below a tmpfs-covered directory."""
    relative = target.relative_to(root)
    current = root
    for part in relative.parts:
        current /= part
        if current not in made:
            argv.extend(("--dir", str(current)))
            made.add(current)


def _sandbox_scratch_path(workdir: Path) -> Path:
    work = workdir.resolve()
    digest = hashlib.sha256(str(work).encode()).hexdigest()[:24]
    return Path("/tmp/oasis-cell-scratch") / digest


def _relocate_dune_cache(dune_cache: Path) -> None:
    """Point copied DUNE compiler commands at the cache's sandbox mount."""
    script = dune_cache / "python" / "dune" / "generated" / "buildScript.sh"
    if not script.is_file():
        raise RuntimeError(
            f"neutral DUNE baseline has no generated compiler script: {script}")
    text = script.read_text()
    prefix_line = next(
        (line.strip() for line in text.splitlines()
         if line.strip().startswith("DUNE_CXX_COMPILER_LAUNCHER=")
         and line.strip().endswith("/compiler_launcher.sh")),
        None)
    if prefix_line is None:
        raise RuntimeError(
            "neutral DUNE baseline compiler script has no cache-root marker")
    old_root = prefix_line.split("=", 1)[1].removesuffix(
        "/compiler_launcher.sh")
    sandbox_root = "/tmp/dune-cache/dune-py"
    launchers = (script, dune_cache / "compiler_launcher.sh")
    for launcher in launchers:
        if not launcher.is_file():
            raise RuntimeError(
                f"neutral DUNE baseline is missing launcher: {launcher}")
        content = launcher.read_text()
        launcher.write_text(content.replace(old_root, sandbox_root))
        if old_root in launcher.read_text():
            raise RuntimeError(
                f"DUNE launcher still references baseline path: {launcher}")


def sandbox_scratch_for(workdir: Path) -> Path:
    """Return the host scratch visible as ``/tmp`` to exactly one cell."""
    root = Path("/tmp/oasis-cell-scratch")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    scratch = _sandbox_scratch_path(workdir)
    scratch.mkdir(mode=0o700, exist_ok=True)
    _OWNED_SCRATCH.add(scratch)
    baseline = os.environ.get("OASIS_DUNE_CACHE_BASELINE")
    dune_cache = scratch / "dune-cache" / "dune-py"
    if baseline and not dune_cache.exists():
        dune_cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(Path(baseline), dune_cache, symlinks=True)
        _relocate_dune_cache(dune_cache)
    return scratch


def cleanup_sandbox_scratch(workdir: Path) -> None:
    """Remove disposable caches after the cell's processes have exited."""
    scratch = _sandbox_scratch_path(workdir)
    shutil.rmtree(scratch, ignore_errors=True)
    _OWNED_SCRATCH.discard(scratch)


def _sandboxed_process_argv(workdir: Path, process: list[str], *,
                            source_repo: Path | None = None) -> list[str]:
    """Run a process with only this cell and private scratch writable."""
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise RuntimeError(
            "bubblewrap is required for blind-run shell isolation")

    work = workdir.resolve()
    private_tmp = sandbox_scratch_for(work)
    home = Path.home().resolve()
    workspace = REPO.parent.resolve()
    runtime_paths = (
        workspace / "open-fem-agent/.venv",
        workspace / "febio-src/cbuild",
        workspace / "sparta/src",
        workspace / "sparta/data",
        workspace / "sparta/examples",
        home / "miniconda3",
        home / "4C",
        home / "FEBio",
        home / "dealii",
        home / ".local/share/python",
        home / ".local/include",
        home / ".local/lib",
    )

    argv = [
        bwrap,
        "--die-with-parent",
        "--unshare-pid",
        "--unshare-ipc",
        "--ro-bind", "/", "/",
        "--dev-bind", "/dev", "/dev",
        "--proc", "/proc",
        "--tmpfs", "/dev/shm",
        # Hide user credentials, every other checkout, and historical runs.
        # Only the explicit solver runtime roots below are rebound.
        "--tmpfs", str(home),
    ]
    made: set[Path] = set()
    for runtime in runtime_paths:
        if runtime.is_dir():
            _add_mount_dirs(argv, home, runtime, made)
            argv.extend(("--ro-bind", str(runtime), str(runtime)))
    dune_host_cache = home / "miniconda3/envs/dune-py313/.cache"
    if dune_host_cache.is_dir():
        argv.extend(("--tmpfs", str(dune_host_cache)))

    # `/tmp` persists between this cell's shell calls, but maps back inside the
    # cell rather than to the host's shared scratch tree.
    argv.extend(("--bind", str(private_tmp), "/tmp"))

    for hidden_root in (workspace, Path("/tmp")):
        if work.is_relative_to(hidden_root):
            _add_mount_dirs(argv, hidden_root, work, made)
            break
    argv.extend(("--bind", str(work), str(work)))

    cwd = work
    if source_repo is not None:
        source = source_repo.resolve()
        source_mount = Path("/tmp/oasis-source")
        _add_mount_dirs(argv, Path("/tmp"), source_mount, made)
        argv.extend(("--ro-bind", str(source), str(source_mount)))
        sessions = source / "data" / "sessions"
        if sessions.is_dir():
            session_output = work / ".oasis_sessions"
            session_output.mkdir(parents=True, exist_ok=True)
            argv.extend(("--bind", str(session_output),
                         str(source_mount / "data" / "sessions")))
        cwd = source_mount / "src"

    argv.extend((
        "--setenv", "HOME", str(work),
        "--setenv", "TMPDIR", "/tmp",
        "--setenv", "XDG_CACHE_HOME", "/tmp/.cache",
        "--setenv", "PYTHONPYCACHEPREFIX", "/tmp/pycache",
        "--setenv", "DUNE_PY_DIR", "/tmp/dune-cache",
        "--chdir", str(cwd),
    ))
    argv.extend(process)
    return argv


def _sandboxed_bash_argv(workdir: Path, command: str) -> list[str]:
    """Build the isolated argv for one host shell call."""
    return _sandboxed_process_argv(
        workdir, ["/bin/bash", "-lc", command])


def _bash_tool_for(workdir: Path, *, audit_on_submit: bool = False):
    # THE AUTO-AUDIT WAS ATTACHED TO write_file ONLY, AND AGENTS SUBMIT WITH A
    # HEREDOC.
    #
    # Its docstring says the write of RESULT.txt is "the only moment that
    # reaches 100% of submitters". Measured over the OASiS-arm runs whose
    # trajectory records the write at all: 57% wrote RESULT.txt by SHELL only,
    # 29% by both, 14% by write_file only — and an auto-audit reply appears in
    # 29% of them. So the hook reached about a quarter of submitters, not all.
    #
    # C1_27b_MCP_seed84 is the case in full. It wrote RESULT.txt with
    # write_file into a nested work/work/ directory, where the audit fired and
    # correctly reported "found NO per-level result files to check"; then it
    # ran `rm -rf work` and wrote the real submission with a run_bash heredoc,
    # which no audit watches. Its residual history is 0.5*0.5^k, bit-identical
    # at all three levels, and the audit refuses exactly that when it is given
    # the chance — every check fires when run by hand.
    #
    # So the same audit now also runs after a shell command that TOUCHED
    # RESULT.txt. Nothing is forced and nothing is blocked: the findings are
    # appended to the reply the agent is already reading, at the moment the
    # submission exists.
    def _audit_after_shell(before: float | None) -> str:
        if not audit_on_submit:
            return ""
        try:
            rt = next(iter(sorted(workdir.rglob("RESULT.txt"))), None)
            if rt is None:
                return ""
            if before is not None and rt.stat().st_mtime <= before:
                return ""            # untouched by this command
            findings = _audit_submission(rt, rt.read_text(errors="replace"))
        except Exception as exc:                       # noqa: BLE001
            return f"\n[auto-audit unavailable: {type(exc).__name__}]"
        return _format_audit_reply(findings)

    # THE PER-LEVEL ARTEFACTS ARE WRITTEN BY THE AGENT'S OWN SOLVER SCRIPTS,
    # WHICH NO write_file HOOK CAN SEE.
    #
    # The early artefact check was attached to write_file and reached NONE of
    # round 7's three OASiS runs, while the submission audit reached all three.
    # Measured in their work dirs: 6, 3 and 11 Python scripts producing 12, 5
    # and 15 per-level CSVs. The agent writes a program with write_file and the
    # PROGRAM writes the deliverables, so the only channel that sees them is
    # the shell command that ran it. This is the same lesson this file already
    # records for RESULT.txt -- 57% of submitters wrote it by shell only -- and
    # the fix there was never extended to the artefacts.
    _ART = ("residual_level*.csv", "interface_level*_[AB].csv",
            "solution_level*.csv")
    # A PARTICIPANT SCRIPT WRITTEN BY HEREDOC IS STILL A PARTICIPANT SCRIPT.
    # 18 of 18 runs on the coupled cell set FACE_HEAT_FLUX with no condition;
    # catching that only on write_file would miss every agent that uses a
    # heredoc, which this file already measured at 57% for RESULT.txt.
    _SCRIPTS = ("*.py",)

    def _artefact_mtimes() -> dict:
        out = {}
        for pat in _ART:
            for f in workdir.rglob(pat):
                try:
                    out[f] = f.stat().st_mtime
                except OSError:
                    pass
        return out

    def _script_mtimes() -> dict:
        out = {}
        for pat in _SCRIPTS:
            for f in workdir.rglob(pat):
                try:
                    out[f] = f.stat().st_mtime
                except OSError:
                    pass
        return out

    def _script_check_after_shell(before: dict) -> str:
        if not audit_on_submit:
            return ""
        try:
            now = _script_mtimes()
            touched = [f for f, t in now.items()
                       if before.get(f) is None or t > before[f]]
            for f in sorted(touched, key=lambda x: now[x], reverse=True):
                try:
                    _txt = f.read_text(errors="replace")
                    got = (_script_noop_check(f, _txt)
                           + _extra_script_checks(f, _txt))
                except OSError:
                    continue
                if got:
                    return got            # one finding per command, not per file
        except Exception:                 # noqa: BLE001
            return ""
        return ""

    def _artefact_check_after_shell(before: dict) -> str:
        if not audit_on_submit:
            return ""
        try:
            now = _artefact_mtimes()
            touched = [f for f, t in now.items()
                       if before.get(f) is None or t > before[f]]
            if not touched:
                return ""
            # ONE CHECK PER KIND, not one check overall.
            #
            # The first version took the single newest touched file. A solver
            # script writes every artefact in one go, so "newest" is arbitrary
            # within the batch: measured on a nine-file write it picked
            # residual_level1.csv, whose history was fine, and the INTERFACE
            # SIGN -- the finding that decided the round -- was never looked
            # at. Grouping by kind bounds the output at two blocks while
            # making sure each check that the batch made possible is run.
            blocks = []
            for pat in _ART:
                import fnmatch
                same = [f for f in touched if fnmatch.fnmatch(f.name, pat)]
                if not same:
                    continue
                newest = max(same, key=lambda f: now[f])
                got = _early_artefact_check(workdir, newest)
                if got:
                    blocks.append(got)
            return "".join(blocks)
        except Exception:                              # noqa: BLE001
            return ""

    def _result_mtime() -> float | None:
        try:
            rt = next(iter(sorted(workdir.rglob("RESULT.txt"))), None)
            return rt.stat().st_mtime if rt else None
        except Exception:                              # noqa: BLE001
            return None

    @tool
    def run_bash(command: str) -> str:
        """Run a shell command inside the cell's sandbox dir. Returns stdout+stderr (truncated to 12 KB)."""
        note_action()
        _before = _result_mtime()
        _before_art = _artefact_mtimes()
        _before_scr = _script_mtimes()
        # THE WHOLE PROCESS GROUP DIES ON TIMEOUT, NOT JUST THE SHELL.
        #
        # This was subprocess.run(..., timeout=900). On timeout Python kills
        # its direct child — which is `bash` — and every GRANDCHILD survives,
        # reparented to init. A solver launched from that shell then runs
        # forever: three DUNE processes from round-1 timeouts were found
        # 23-24 hours later, still at 100% CPU, having burned 71 CPU-hours
        # between them. They also contend for the machine with whatever runs
        # next, which silently slows and can time out later cells.
        #
        # start_new_session puts the shell and everything it spawns in one
        # process group; on timeout that group is signalled, TERM then KILL.
        proc = None
        try:
            proc = subprocess.Popen(
                _sandboxed_bash_argv(workdir, command), cwd=workdir,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                env=_clean_subprocess_env(),
                start_new_session=True,
            )
            out_s, err_s = proc.communicate(timeout=900)
            out = (out_s or "") + (("\n[stderr]\n" + err_s) if err_s else "")
            out = out[-12000:] if len(out) > 12000 else out
            # A submission written by heredoc is still a submission.
            return (out + _script_check_after_shell(_before_scr)
                    + _artefact_check_after_shell(_before_art)
                    + _audit_after_shell(_before) + _time_left_note())
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            return ("[timeout after 900s; the command and everything it "
                    "spawned were terminated]" + _time_left_note())
        except (OSError, UnicodeError, ValueError) as e:
            _kill_group(proc)
            return f"[command failed to launch: {type(e).__name__}: {e}]"
    return run_bash


def _kill_group(proc) -> None:
    """TERM then KILL the process group `proc` leads. Never raises."""
    if proc is None or proc.poll() is not None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            return
        try:
            proc.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            continue


def _read_write_tools_for(workdir: Path, *, audit_on_submit: bool = False):
    @tool
    def read_file(path: str, max_bytes: int = 200_000) -> str:
        """Read a file inside the cell sandbox."""
        try:
            p = Path(path)
            if not p.is_absolute():
                p = workdir / p
            wd = workdir.resolve()
            resolved = p.resolve()
            if not resolved.is_relative_to(wd):
                return (f"[read refused: {p} is outside your working "
                        f"directory {wd}]")
            data = resolved.read_bytes()[:max_bytes]
            return data.decode("utf-8", errors="replace")
        except FileNotFoundError:
            return f"[file not found: {path}]"
        except (OSError, ValueError) as e:
            return f"[read failed: {type(e).__name__}: {e}]"

    @tool
    def write_file(path: str, content: str) -> str:
        """Write `content` to `path` (relative paths resolve inside the cell sandbox)."""
        try:
            p = Path(path)
            if not p.is_absolute():
                p = workdir / p
            wd = workdir.resolve()
            if not p.resolve().is_relative_to(wd):
                return (f"[write refused: {p} is outside your working "
                        f"directory {wd}. All files — scripts, logs, and "
                        f"every required deliverable — must be written "
                        f"inside it; use a relative path.]")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            reply = f"wrote {len(content)} chars to {p}"
            # ROUND-8 MECHANISM, OASiS ARM ONLY: the submission is audited the
            # moment it is written, and the findings are placed in the reply
            # the agent is already reading. Round 7 measured why voluntary
            # does not work: a calibrated audit tool plus the instruction to
            # run it, sitting in the one channel read by every run, was used
            # by 1 of 51 agents — while the tool's checks catch 15 of the 18
            # submitted-and-wrong runs of earlier rounds. Participation is now
            # by default. The bare arm is untouched: this flag is set only by
            # build_mcp_agent, because the audit is OASiS's capability.
            # FIRE THE COUPLED CHECKS ON THE ARTEFACT WRITE, not only on
            # submission. See _early_artefact_check for the mtime measurement
            # that forced this.
            if audit_on_submit and p.name != "RESULT.txt":
                reply += _script_noop_check(p, content)
                reply += _extra_script_checks(p, content)
                reply += _early_artefact_check(workdir, p)
            if audit_on_submit and p.name == "RESULT.txt":
                # A GIVE-UP FILED OVER FINISHED WORK, caught structurally.
                # Checked before the numeric audit because it is the more
                # basic error: the audit asks whether the numbers are
                # self-consistent, this asks whether numbers were submitted
                # at all when they existed.
                if "COULD_NOT_COMPLETE" in content.upper():
                    try:
                        _contra = _work_on_disk_contradicting_a_give_up(workdir)
                    except Exception:                # noqa: BLE001
                        _contra = ""
                    if _contra:
                        reply += "\n\n" + _contra
                try:
                    findings = _audit_submission(p, content)
                except Exception as e:               # noqa: BLE001
                    findings = None
                    reply += f"\n[auto-audit unavailable: {type(e).__name__}]"
                # ORDER MATTERS: "NOEVIDENCE" IS A TRUTHY STRING.
                #
                # This chain tested `if findings:` first, so the sentinel took
                # the findings branch and the agent received the bare token
                # NOEVIDENCE under the heading "AUTO-AUDIT of your submission",
                # followed by "check the named place, fix if real". The
                # paragraph below — written precisely because the old code told
                # 134 of 137 HONEST_INCOMPLETE runs "clean" — could never
                # print. HONEST_INCOMPLETE is the largest single bucket in the
                # campaign at 159 rows, so the message that never printed was
                # the one aimed at the most common outcome.
                # NEVER an all-clear on nothing, and never two copies of the
                # wording: _format_audit_reply is shared with the shell route.
                reply += _format_audit_reply(findings)
            return reply
        except (OSError, UnicodeError, ValueError) as e:
            return (f"[write failed: {type(e).__name__}: {e} — "
                    "use a relative path inside the sandbox]")

    return [read_file, write_file]


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web (DuckDuckGo). Returns up to max_results result snippets.

    DuckDuckGo occasionally rate-limits the API backend; we transparently
    try its ``html`` and ``lite`` backends as fallbacks so the tool stays
    useful through brief blocks.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return ("[web_search unavailable: install duckduckgo-search "
                "(pip install duckduckgo-search) to enable]")
    last_err = None
    for backend in ("auto", "html", "lite"):
        try:
            with DDGS() as ddgs:
                hits = list(ddgs.text(query, max_results=max_results,
                                      backend=backend))
            if hits:
                return "\n\n".join(
                    f"{h.get('title')}\n{h.get('href')}\n{h.get('body')}"
                    for h in hits)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            continue
    return f"[no results; last backend error: {last_err}]" if last_err \
        else "[no results]"


# ────────────────────────────────────────────────────────────────────
# spawn_subagent — sibling LangGraph agent on the same vLLM endpoint
# ────────────────────────────────────────────────────────────────────
def _make_spawn_subagent_tool(
    *, size: str, seed: int, workdir: Path,
    parent_tools: Sequence[BaseTool], depth: int,
):
    """Returns a tool that spawns a depth-limited sibling agent.

    The sub-agent reuses the same vLLM server (cheap on memory) with a
    slightly higher temperature and a derived seed. It gets the same
    workdir-bound bash/read/write/web_search and the parent's OASiS
    tools, but its own ``spawn_subagent`` is *not* re-installed beyond
    depth 1 to prevent runaway recursion.
    """

    @tool
    async def spawn_subagent(role: str, task: str, context: str = "") -> str:
        """Spawn a sub-agent. role∈{critic, researcher, verifier}; task = what it should do; context = facts to pass in.

        The critic role should ruthlessly challenge the parent's setup; the
        verifier should re-derive numbers independently; the researcher
        should look things up via web_search and the OASiS knowledge tool.
        Returns the sub-agent's final message text.
        """
        # ASYNC, AND ainvoke BELOW, BECAUSE THE MCP TOOLS ARE ASYNC-ONLY.
        # This was a sync `def` calling `sub_agent.invoke`. The sub-agent
        # inherits the parent's OASiS tools, which langchain_mcp_adapters
        # returns as coroutine-only StructuredTools, so the first time a
        # critic reached for `knowledge` or `discover` it raised
        # "NotImplementedError: StructuredTool does not support sync
        # invocation" — caught by the except below and returned to the model
        # as a string, so it looked like a critic verdict rather than a dead
        # mechanism. Round 1: this fired in 14 of 14 coupled and 16 of 18
        # single-code OASiS runs, i.e. the MANDATORY critic the server
        # instructions demand never ran once in the entire campaign.
        if depth >= 2:
            return "[spawn_subagent denied: max depth 2 to prevent recursion]"
        sub_tools = list(parent_tools)
        # Give the sub-agent the same host tools, but no further nesting:
        sub_tools = [t for t in sub_tools if t.name != "spawn_subagent"]
        if role == "critic":
            sys = (
                "You are a ruthlessly critical reviewer. Challenge every "
                "parameter choice, check units, look for sign errors, "
                "verify BCs, and validate against the literature via "
                "web_search. Respond with one of: APPROVED: <reason> | "
                "REJECTED: <issue and required fix>."
            )
        elif role == "verifier":
            sys = (
                "You are an independent verifier. Re-derive the requested "
                "quantity from first principles or by an alternative "
                "method/solver, then compare with the parent's number."
            )
        else:
            sys = (
                "You are a research assistant. Look up authoritative "
                "sources for the requested information and summarise."
            )
        sub_llm = _llm(size, temperature=0.3, seed=seed + 100 + depth)
        sub_agent = create_react_agent(
            sub_llm, tools=sub_tools, prompt=sys,
        )
        msg = f"Task: {task}\n\nContext provided by parent:\n{context}"
        try:
            out = await sub_agent.ainvoke(
                {"messages": [("user", msg)]},
                config={"recursion_limit": 40},
            )
            return out["messages"][-1].content
        except Exception as e:
            # Still returned as text so one bad sub-agent cannot kill the run,
            # but marked loudly enough that a transcript sweep finds it: a
            # broken mechanism must not read like a verdict.
            return (f"[SUBAGENT FAILED — this is NOT a review verdict — "
                    f"{type(e).__name__}: {e}]")

    return spawn_subagent


# ────────────────────────────────────────────────────────────────────
# OASiS MCP tool loader (langchain-mcp-adapters)
# ────────────────────────────────────────────────────────────────────
def _oasis_mcp_client(workdir: Path | None = None):
    from langchain_mcp_adapters.client import MultiServerMCPClient

    env = _clean_subprocess_env()
    env.pop("OFA_DISABLE_CRITIC", None)
    env.pop("OFA_DISABLE_PITFALLS", None)
    env["FOURC_ROOT"] = env.get("FOURC_ROOT", str(Path.home() / "4C"))
    env["FOURC_BINARY"] = env.get(
        "FOURC_BINARY", str(Path.home() / "4C/build/4C"))
    # LD_LIBRARY_PATH CARRIES MORE THAN ONE SOLVER, AND `get(default)` DROPS
    # THE REST.
    #
    # This was env.get("LD_LIBRARY_PATH", "/opt/4C-dependencies/lib"): if the
    # outer environment had the variable set to anything at all, 4C's
    # dependency path was silently discarded, and preCICE's was never added
    # under any circumstance. Measured on this host: `import precice` fails
    # with "libprecice.so.3: cannot open shared object file" although
    # /opt/precice/lib/libprecice.so.3 -> libprecice.so.3.1.2 is present and
    # the Python binding is installed, and it succeeds the moment
    # /opt/precice/lib is on the path (returns 3.1.2;v3.1.2). 103 of the C2
    # run directories mention preCICE, so agents do reach for that path and
    # it could not load for any of them.
    #
    # Composed rather than defaulted: every required entry, then whatever was
    # inherited, order preserved, duplicates dropped, and entries that do not
    # exist on this host left out so a stale path cannot mask a real one.
    _lib_dirs = ["/opt/4C-dependencies/lib", "/opt/precice/lib"]
    _seen, _parts = set(), []
    for _d in _lib_dirs + [
            x for x in env.get("LD_LIBRARY_PATH", "").split(":") if x]:
        if _d in _seen:
            continue
        _seen.add(_d)
        if Path(_d).is_dir():
            _parts.append(_d)
    env["LD_LIBRARY_PATH"] = ":".join(_parts)
    _pin_backend_runtime_env(env)
    source_repo = Path(os.environ.get(
        "OASIS_SOURCE_SNAPSHOT", str(REPO))).resolve()
    env["PYTHONPATH"] = str(source_repo / "src")
    # THE TOOLS MUST WRITE INTO THIS CELL'S SANDBOX.
    #
    # run_simulation and friends wrote to <repo>/simulation_outputs, a single
    # directory shared by every caller. An agent that followed the documented
    # OASiS workflow therefore produced its solution files where nothing
    # downstream looks, and where another cell could overwrite or read them.
    # It is the OASiS-arm tools that do this, so the cost fell entirely on the
    # arm under test: 8 of 14 coupled OASiS runs in round 1 went through it.
    if workdir is not None:
        cell_work = Path(workdir).resolve()
        env["OASIS_CELL_WORKDIR"] = str(cell_work)
        env["OASIS_OUTPUT_DIR"] = str(cell_work / "simulation_outputs")
        env["OASIS_COUPLING_DIR"] = str(cell_work / "coupling")
        env["OASIS_MESH_DIR"] = str(cell_work / "meshes")
        env["OASIS_BENCHMARK_DIR"] = str(cell_work / "benchmark_results")

    # THE SERVER INTERPRETER, RESOLVED — NOT ASSUMED.
    #
    # This was REPO/".venv/bin/python". A git worktree has no .venv, so on any
    # worktree checkout every MCP-arm run crashed at agent construction with
    # FileNotFoundError before the model was ever called — while the BARE arm
    # ran fine. A fleet launched that way silently becomes one-armed, which is
    # the worst possible shape for an A/B campaign. Third instance of this
    # exact defect class (backend_imports.json, the fixture runner) — same
    # fix: explicit env var first, then the repo venv, then the primary
    # checkout's venv, and REFUSE loudly rather than launch a crippled arm.
    _cands = [os.environ.get("OASIS_PYTHON"),
              str(REPO / ".venv/bin/python"),
              str(Path.home() / "Schreibtisch/open-fem-agent/.venv/bin/python")]
    _server_py = next((c for c in _cands if c and Path(c).is_file()), None)
    if _server_py is None:
        raise RuntimeError(
            "no interpreter found for the OASiS MCP server; set OASIS_PYTHON. "
            "Refusing to build a silently crippled MCP arm.")
    command = _server_py
    args = ["-m", "server"]
    cwd = source_repo / "src"
    if workdir is not None:
        wrapped = _sandboxed_process_argv(
            Path(workdir), [command, *args], source_repo=source_repo)
        command, args = wrapped[0], wrapped[1:]
        cwd = Path(workdir)
        env["PYTHONPATH"] = "/tmp/oasis-source/src"
    client = MultiServerMCPClient({
        "oasis": {
            "command": command,
            "args": args,
            "cwd": str(cwd),
            "env": env,
            "transport": "stdio",
        }
    })
    return client


def _load_oasis_mcp_tools(workdir: Path | None = None) -> list[BaseTool]:
    """List tools for discovery-only callers.

    Agent runs must use :func:`oasis_mcp_tools_session`; tools returned here
    create a new server for every call and therefore cannot carry server-side
    state such as critic reviews.
    """
    client = _oasis_mcp_client(workdir)
    return asyncio.run(client.get_tools())


@asynccontextmanager
async def oasis_mcp_tools_session(workdir: Path | None = None):
    """Yield tools bound to one OASiS process for an entire agent run."""
    from langchain_mcp_adapters.tools import load_mcp_tools

    client = _oasis_mcp_client(workdir)
    async with client.session("oasis") as session:
        available = await load_mcp_tools(session, server_name="oasis")
        names = {tool.name for tool in available}
        missing = CAMPAIGN_MCP_TOOL_ALLOWLIST - names
        if missing:
            raise RuntimeError(
                "OASiS campaign tool contract is incomplete; missing: "
                + ", ".join(sorted(missing)))
        yield [tool for tool in available
               if tool.name in CAMPAIGN_MCP_TOOL_ALLOWLIST]


# ────────────────────────────────────────────────────────────────────
# Public factories
# ────────────────────────────────────────────────────────────────────


# sys.path is extended ONCE here, not on every RESULT.txt write —
# the per-call insert accumulated 27 duplicate entries in 25 writes
# and kept putting OASiS's src ahead of the venv for every import.
import sys as _sys_for_path
_sys_for_path.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src"))


def _flat(v):
    """Every scalar in a nested list, however the participant shaped it."""
    if isinstance(v, (list, tuple)):
        for x in v:
            yield from _flat(x)
    elif isinstance(v, (int, float)):
        yield v


def _work_on_disk_contradicting_a_give_up(work: Path) -> str:
    """A give-up written on top of a finished run — reported structurally.

    Measured twice. Ten coupled runs drove a coupling to convergence, hit a
    flux-balance finding, and declared COULD_NOT_COMPLETE with a median 69% of
    their budget unspent. The tool's reply was then rewritten to say NOT
    VERIFIED and NOT A RESULT are different verdicts and to write the
    deliverables first — and in the very next probe three of six runs did it
    again anyway (C7, C8, C9), one of them stating in its own words that the
    coupling converged in ~7 iterations before giving up on the balance check.

    Wording the imperative better does not work; the same lesson as the audit
    tool that was called by 1 of 51 runs when merely offered. So this reads the
    agent's OWN files at the moment it writes a give-up and says what is
    already there. It supplies no knowledge, no method and no numbers — it only
    refuses to let a finished run be filed as an unfinished one silently.
    """
    import csv as _csv
    sol = sorted(work.rglob("solution_level*.csv"))
    iface = sorted(work.rglob("interface_level*.csv"))
    resid = sorted(work.rglob("residual_level*.csv"))
    # A PARTICIPANT'S OWN EXPORT COUNTS AS WORK. C8 of the seed-13 probe wrote
    # no CSV at all and still had exports.json for both halves of level 1 —
    # a solve that ran and an interface exchange that completed, filed as
    # COULD_NOT_COMPLETE. Looking only for the task's deliverables misses
    # exactly the run that did the work and never wrote it down.
    # Non-empty AND not the iteration-1 fallback: a participant that wrote
    # only placeholder zeros has not solved anything, and calling that "work
    # on disk" would be the same crying-wolf that teaches agents to ignore a
    # gate. Checked on the four seed-13 give-ups: all four carry real numbers
    # (|q| up to 0.98), so none of them is a false positive.
    def _real(q):
        try:
            import json as _json
            d = _json.loads(q.read_text())
        except Exception:                            # noqa: BLE001
            return False
        for key in ("values", "normal_fluxes"):
            for x in _flat(d.get(key) or []):
                if x not in (0, 0.0) and x == x:
                    return True
        return False

    exports = [q for q in sorted(work.rglob("exports.json")) if _real(q)]
    if not (sol or iface or resid or exports):
        return ""                      # nothing on disk: the give-up is honest

    conv = []
    for r in resid:
        try:
            rows = [row for row in _csv.reader(r.open()) if row]
            vals = []
            for row in rows[1:]:
                try:
                    vals.append(float(row[-1]))
                except (ValueError, IndexError):
                    pass
            if vals:
                conv.append((r.name, len(vals), vals[-1]))
        except OSError:
            pass

    bits = []
    if sol:
        bits.append(f"{len(sol)} solution_level*.csv")
    if iface:
        bits.append(f"{len(iface)} interface_level*.csv")
    if exports and not (sol or iface):
        bits.append(f"{len(exports)} participant exports.json — a solve ran "
                    f"and an interface exchange completed, but none of the "
                    f"task's own output files were written")
    for name, n, last in conv:
        bits.append(f"{name} with {n} iterations ending at {last:.3g}")
    return (
        "YOU ARE FILING A GIVE-UP ON TOP OF WORK THAT IS ON DISK.\n  "
        + "\n  ".join(bits)
        + "\nCOULD_NOT_COMPLETE is graded as nothing. A submission built from "
          "the numbers you already have is graded on those numbers, PROVIDED "
          "it is complete: every level the task prescribes and, where the "
          "task names two subdomains, both files per level. A submission "
          "missing a level or a side is malformed and scores the same zero "
          "as no submission, so complete the sequence from what you have "
          "rather than filing part of it. A verification "
          "finding — a flux imbalance, a failed conservation check — is NOT a "
          "reason to withhold a field your solve already produced: those are "
          "different verdicts and only one of them is worth zero. Write the "
          "deliverables the task asks for from what you have, state the "
          "finding alongside them, and keep working on the finding with "
          "whatever time is left."
    )


def _script_noop_check(written: Path, content: str) -> str:
    """A participant that sets a nodal flux and creates no condition is inert.

    MEASURED, and this is the reason this check exists rather than another
    paragraph of advice. Over the 18 OASiS runs of this cell served the fact:
    18 of 18 called a knowledge door, 18 of 18 set FACE_HEAT_FLUX, and ZERO of
    18 created the condition that makes it do anything. They find OASiS, they
    read it, they get the concept, and the one line that turns a nodal value
    into a boundary condition does not survive into the code.

    The consequence is invisible at runtime: Kratos converges, exits 0, and
    returns exactly the no-flux field -- 2.307291e-03 against 3.605675e-03
    with the condition present, bit-identical to a zero-flux run.

    So it is caught in the SCRIPT, when the script is written, before it has
    run once. Prose next to the fact did not work eighteen times.
    """
    import re as _re

    if written.suffix != ".py":
        return ""
    sets_flux = "FACE_HEAT_FLUX" in content
    if not sets_flux:
        return ""
    has_cond = bool(_re.search(
        r"CreateNewCondition\s*\(\s*[\"']"
        r"(ThermalFace\dD\dN|FluxCondition\dD\dN)", content))
    if has_cond:
        return ""
    return _FLUX_NOOP_MSG(written)


def _extra_script_checks(written: Path, content: str) -> str:
    """Two more defects that are visible in the script and invisible at runtime.

    Both were reproduced by execution, and both are counted in the campaign's
    own OASiS-arm scripts (per file, so a correct usage elsewhere cannot excuse
    a broken one here):

      DUNE `solver="cg"` on an operator carrying advection -- 20 runs. cg is
        accepted on a NON-SYMMETRIC operator and scheme.solve does not raise:
        it returns converged=False, linear_iterations=-10000, and leaves the
        field at the initial guess, so every level is exactly zero. Measured:
        cg gave peak 0.000000e+00 at all three levels, bicgstab 8.875850e-02.
        67 further runs use cg with no advection term, which is defensible, so
        the check requires the advection.

      NGSolve x/y rebound before symbolic use -- 27 runs. After
        `from ngsolve import *`, any loop assigning x or y rebinds the
        symbolic coordinates to floats. Measured: type() goes
        CoefficientFunction -> float with value 0.02514662, x and y left at
        0.9886363636, and CoefficientFunction((float, float)) is accepted
        silently. A constant body force on a fully-Dirichlet incompressible
        domain gives u identically zero -- 7.16e-17, 3.60e-17, 1.30e-17,
        order 0.0000 -- against 1.2229e-02 with the symbolic source.
    """
    import re as _re

    if written.suffix != ".py":
        return ""
    out = []
    if _re.search(r"solver\s*=\s*[\"']cg[\"']", content) and _re.search(
            r"dot\s*\(\s*b\w*\s*,\s*grad|inner\s*\(\s*b\w*\s*,\s*grad"
            r"|velocity|\badvect", content, _re.I):
        out.append(
            "  * THIS SCRIPT USES solver=\"cg\" ON AN OPERATOR WITH AN "
            "ADVECTION TERM, which is not symmetric. cg is ACCEPTED anyway and "
            "scheme.solve DOES NOT RAISE: it returns converged=False, "
            "linear_iterations=-10000, and leaves the field at the INITIAL "
            "GUESS, so every level comes out exactly zero with exit 0. "
            "Measured: cg gave peak 0.000000e+00 at all three levels against "
            "8.875850e-02 for bicgstab. Use bicgstab, or gmres WITH an "
            "assertion on info['converged'], or a direct solver -- and assert "
            "peak|u| > 0 at every level.")
    if ("from ngsolve import *" in content or "import ngsolve" in content):
        m = _re.search(r"^\s*(?:x|y)\s*=\s*\(?\s*i\w*\s*\+\s*0\.5",
                       content, _re.M)
        if m and _re.search(
                r"(?:sin|cos|exp)\s*\(\s*[^)]*\b[xy]\b|CoefficientFunction"
                r"|grad\s*\(|GridFunction", content[m.end():]):
            out.append(
                "  * THIS SCRIPT REBINDS `x` OR `y` IN A LOOP AND THEN USES "
                "THEM SYMBOLICALLY. After `from ngsolve import *` those names "
                "ARE the symbolic coordinates, so the assignment turns your "
                "source into a CONSTANT -- measured, type() goes "
                "CoefficientFunction -> float with value 0.02514662 and x, y "
                "left at 0.9886363636, and CoefficientFunction((float, float)) "
                "is accepted silently. A constant body force on a "
                "fully-Dirichlet incompressible domain gives u identically "
                "zero: 7.16e-17, 3.60e-17, 1.30e-17 across the levels, order "
                "0.0000, against 1.2229e-02 with the symbolic source. Name "
                "your probe coordinates px, py -- and print type(source) "
                "before you assemble.")
    if not out:
        return ""
    return ("\n\n[early check of " + written.name + ", read from the script "
            "you just wrote:]\n" + "\n".join(out)
            + "\nYou have budget left now; after the solve this looks like a "
              "converged run.")


def _FLUX_NOOP_MSG(written: Path) -> str:
    return ("\n\n[early check of " + written.name + ", read from the script "
            "you just wrote:]\n"
            "  * THIS SCRIPT SETS FACE_HEAT_FLUX AND CREATES NO CONDITION, so "
            "the flux will be silently discarded. A nodal value is only ever "
            "integrated BY a condition; with none on the interface edges "
            "Kratos runs, converges, exits 0 and returns exactly the field it "
            "would have returned with no flux at all. Measured on one mesh: "
            "zero flux 2.307291e-03, flux on nodes with no condition "
            "2.307291e-03 (BIT-IDENTICAL), flux with conditions 3.605675e-03. "
            "Add, over the interface edges in order:\n"
            "        for c in range(len(iface) - 1):\n"
            "            mp.CreateNewCondition(\"ThermalFace2D2N\", c + 1,\n"
            "                                  [iface[c] + 1, iface[c+1] + 1], "
            "prop)\n"
            "    then set FACE_HEAT_FLUX on those nodes. FluxCondition2D2N "
            "works too. 18 of the last 18 runs on this cell omitted this and "
            "every one of them submitted the no-flux answer.")


def _early_artefact_check(workdir: Path, written: Path) -> str:
    """Check a per-level artefact THE MOMENT IT IS WRITTEN, not at submission.

    WHY, MEASURED. The submission audit is correct, it arrives, and it cannot
    be acted on. File mtimes over six C2 runs of the last two rounds: five of
    them wrote RESULT.txt at 93-99% of their whole file-activity span, with
    only 8 to 115 seconds of activity left afterwards. The one that wrote it
    at 68%, with 357 seconds still to go, is the ONLY one of the six that
    reached a gradeable convergence order with both prescribed codes proven.
    seed301 received SEVEN findings at that moment -- three too-short
    histories, an identical-history-across-levels, near-zero fields -- and had
    40 seconds. Findings delivered with no budget to spend on them change
    nothing.

    So the coupled checks fire on the artefact write. Each one runs ONLY the
    check its own file makes possible, so this costs the agent no actions and
    adds no noise to unrelated writes:

        residual_level<k>.csv        -> is this a history at all? how long,
                                        and is it identical to another level's?
        interface_level<k>_<side>.csv -> once both sides and the matching
                                        solution file exist, the flux SIGN

    OASiS arm only, like every other audit hook: the caller gates it.
    """
    import re as _re

    name = written.name
    try:
        if _re.fullmatch(r"residual_level\d+\.csv", name):
            from tools.result_audit import residual_findings
            seen, found = set(), []
            for f in residual_findings(workdir):
                if not (name in str(f.get("sequence", ""))
                        or "IDENTICAL" in f.get("finding", "")):
                    continue
                # DEDUPE BY TEXT. residual_findings reports per-level, so a
                # three-level submission with the same defect at every level
                # gave the identical sentence three times in one reply.
                key = f.get("finding", "")[:80]
                if key in seen:
                    continue
                seen.add(key)
                found.append(f)
            if found:
                return ("\n\n[early check of " + name + ", from your own file:]\n"
                        + "\n".join(f"  * {f['finding']}" for f in found[:2])
                        + "\nYou have budget left now. Fixing this after "
                          "RESULT.txt is written is usually too late.")
        elif _re.fullmatch(r"solution_level\d+(_[AB])?\.csv", name):
            # THE EXPORT CAN RUIN A PERFECT SOLVE, and the agent can fix it
            # without re-running anything. Proven against the sealed answer:
            # a submission graded CORRECT at order 1.9796 re-exported by
            # nearest-node lookup graded CONFIDENTLY_WRONG at 0.9815, nothing
            # else changed. 99 OASiS-arm runs carry the fingerprint.
            from tools.result_audit import export_findings
            found = [f for f in export_findings(workdir)
                     if name in str(f.get("sequence", ""))]
            if found:
                return ("\n\n[early check of " + name + ", from your own file:]\n"
                        + "\n".join(f"  * {f['finding']}" for f in found[:1])
                        + "\nThis is a POST-PROCESSING fix: you do not need to "
                          "re-run the solver, only to re-read it.")
        elif _re.fullmatch(r"interface_level\d+_[AB]\.csv", name):
            from tools.result_audit import interface_sign_findings
            lvl = _re.search(r"level(\d+)", name).group(1)
            both = all((workdir / f"interface_level{lvl}_{s}.csv").is_file()
                       or list(workdir.rglob(f"interface_level{lvl}_{s}.csv"))
                       for s in ("A", "B"))
            if not both:
                return ""                  # the other side is not written yet
            found = interface_sign_findings(workdir)
            hard = [f for f in found if "WRONG SIGN" in f.get("finding", "")
                    or "DOES NOT SHRINK" in f.get("finding", "")]
            if hard:
                return ("\n\n[early check of the interface at level " + lvl
                        + ", from your own files:]\n"
                        + "\n".join(f"  * {f['finding']}" for f in hard[:2])
                        + "\nYou have budget left now. This is the failure "
                          "that a clean convergence order cannot reveal.")
    except Exception:                      # noqa: BLE001
        return ""
    return ""


def _audit_submission(result_path: Path, content: str):
    """Run the OASiS result audit in-process on the submission's directory.

    Returns a findings string, "" for clean, and raises only when the audit
    module itself is unavailable. claimed order is parsed from the submission
    text so ORDER MISMATCH can fire; absent an order claim, the zero-field,
    floor and non-monotone checks still run.
    """
    import re as _re
    from tools.result_audit import audit as _audit
    # LAST match, line-anchored, sign/exponent allowed, and labels that only
    # LOOK like an order excluded. The old regex took the FIRST match of a
    # loose pattern: "ORDER_OF_MAGNITUDE = 5" became a claim of order 5, and
    # an ELEMENT_ORDER line ahead of the real one won. It also missed
    # ORDER_L2 (a digit ends [A-Z_]*), lowercase, and negatives.
    claimed = None
    for mm in _re.finditer(
            r"^\s*(?!.*OF_MAGNITUDE)([A-Za-z_0-9]*ORDER[A-Za-z_0-9]*)\s*=\s*"
            r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$",
            content, _re.M):
        lab = mm.group(1).upper()
        if any(k in lab for k in ("ELEMENT", "POLYNOMIAL", "DEGREE", "MESH")):
            continue                      # describes the discretisation
        try:
            claimed = float(mm.group(2))
        except ValueError:
            claimed = None
    r = _audit(str(result_path.parent), claimed_order=claimed)
    if r.get("sequences_found", 0) == 0 and r.get("clean"):
        return "NOEVIDENCE"
    if r.get("clean"):
        return ""
    return "\n".join(f"  * {f['sequence']}: {f['finding']}"
                      for f in r.get("findings", []))

def _host_tools(workdir: Path, *, size: str, seed: int,
                parent_tools: list[BaseTool], depth: int,
                audit_on_submit: bool = False) -> list[BaseTool]:
    tools: list[BaseTool] = []
    tools.append(_bash_tool_for(workdir, audit_on_submit=audit_on_submit))
    tools.extend(_read_write_tools_for(workdir, audit_on_submit=audit_on_submit))
    tools.append(web_search)
    spawn = _make_spawn_subagent_tool(
        size=size, seed=seed, workdir=workdir,
        parent_tools=parent_tools + tools, depth=depth,
    )
    tools.append(spawn)
    return tools


def build_bare_agent(*, size: str, seed: int, workdir: Path, depth: int = 0):
    tools = _host_tools(workdir, size=size, seed=seed,
                        parent_tools=[], depth=depth)
    llm = _llm(size, temperature=0.2, seed=seed)
    return create_react_agent(llm, tools=tools, prompt=BARE_SYSTEM)


@asynccontextmanager
async def build_mcp_agent(*, size: str, seed: int, workdir: Path,
                          depth: int = 0):
    """Yield an MCP agent whose tools share one live OASiS server."""
    async with oasis_mcp_tools_session(workdir) as mcp_tools:
        host = _host_tools(workdir, size=size, seed=seed,
                           parent_tools=mcp_tools, depth=depth,
                           audit_on_submit=True)
        llm = _llm(size, temperature=0.2, seed=seed)
        yield create_react_agent(llm, tools=mcp_tools + host,
                                 prompt=MCP_SYSTEM)


__all__ = ["build_bare_agent", "build_mcp_agent",
           "oasis_mcp_tools_session", "CAMPAIGN_MCP_TOOL_ALLOWLIST",
           "sandbox_scratch_for", "cleanup_sandbox_scratch"]
