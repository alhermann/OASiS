"""LangGraph agents for the open-weight ablation.

Two factory functions:

* :func:`build_bare_agent`  — host-side toolset only (bash + web search +
  spawn_subagent). No OASiS MCP. Mirrors what Claude has in v1 BARE.
* :func:`build_mcp_agent`   — same host-side toolset PLUS every OASiS MCP
  tool attached via langchain-mcp-adapters (MCP_FULL semantics).

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

MCP_FULL additionally exposes all OASiS tools (``prepare_simulation``,
``knowledge``, ``discover``, ``examples``, ``developer``, ``generate_mesh``,
``run_simulation``, ``run_with_generator``, ``coupled_solve``,
``transfer_field``, ``visualize``, ``session_insights``,
``rediscover_backends``) — picked up automatically from the running OASiS
server, so any future tool added there is included without code changes
here.

All LLM calls go through ``langchain_openai.ChatOpenAI`` pointed at a local
vLLM server, which surfaces Qwen2.5's native tool-call format through the
OpenAI schema.
"""
from __future__ import annotations

import asyncio
import os
import signal
import subprocess
from pathlib import Path
from typing import Sequence

from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

REPO = Path(__file__).resolve().parents[1]

PORTS = {"7b": 8000, "14b": 8001, "32b": 8002}

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

MCP_SYSTEM = (
    "When you write your final RESULT.txt, it is automatically audited "
    "against your own output files and any findings appear in the write "
    "confirmation - read them, fix what is real, and rewrite the file. "
    "You are connected to the OASiS MCP server with its full toolset "
    "(prepare_simulation, knowledge, discover, examples, developer, "
    "generate_mesh, run_simulation, run_with_generator, coupled_solve, "
    "transfer_field, visualize, session_insights). Follow the OASiS "
    "workflow: discover → prepare_simulation → examples → "
    "run_(simulation|with_generator) → visualize. Use `knowledge` for "
    "physics + pitfalls, `developer` for source/architecture lookups, and "
    "`coupled_solve` / `transfer_field` for cross-code coupling.\n\n"
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


def _time_left_note() -> str:
    """`[clock: N min left of M]`, or nothing when no deadline is set."""
    if _DEADLINE is None:
        return ""
    import time as _t
    left = _DEADLINE[0] - _t.time()
    total = _DEADLINE[1]
    if left <= 0:
        return "\n[clock: budget spent]"
    return (f"\n[clock: {int(left // 60)} min left of "
            f"{int(total // 60)}]")


def _bash_tool_for(workdir: Path):
    @tool
    def run_bash(command: str) -> str:
        """Run a shell command inside the cell's sandbox dir. Returns stdout+stderr (truncated to 12 KB)."""
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
                ["bash", "-lc", command], cwd=workdir,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                start_new_session=True,
            )
            out_s, err_s = proc.communicate(timeout=900)
            out = (out_s or "") + (("\n[stderr]\n" + err_s) if err_s else "")
            out = out[-12000:] if len(out) > 12000 else out
            return out + _time_left_note()
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
        """Read a file (absolute path, or relative to the cell sandbox)."""
        try:
            p = Path(path)
            if not p.is_absolute():
                p = workdir / p
            data = p.read_bytes()[:max_bytes]
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
                if findings:
                    reply += (
                        "\n\nAUTO-AUDIT of your submission (from your own "
                        "files only — no reference solution):\n" + findings +
                        "\nA finding is a pointer, not a verdict: check the "
                        "named place, fix if real, and REWRITE this file. "
                        "Submitting with a standing finding usually grades "
                        "wrong.")
                elif findings == "":
                    reply += ("\n[auto-audit: clean — self-consistent, which "
                              "is necessary but not sufficient for correct]")
                elif findings == "NOEVIDENCE":
                    # NEVER an all-clear on nothing. Replaying 494 graded runs,
                    # the old code told 134 of 137 HONEST_INCOMPLETE runs
                    # "clean" — because a work dir with no per-level files
                    # produces no sequences, which it read as no findings. The
                    # all-clear went almost exclusively to runs that were NOT
                    # correct, at the moment the agent decides whether to keep
                    # working, in the measured arm only.
                    reply += ("\n[auto-audit: found NO per-level result files "
                              "to check. This is NOT a clean bill — it means "
                              "there is nothing here to verify. If you have "
                              "results, write them per level; if you do not, "
                              "this submission has no numbers behind it.]")
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
def _load_oasis_mcp_tools(workdir: Path | None = None) -> list[BaseTool]:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    env = os.environ.copy()
    env.pop("OFA_DISABLE_CRITIC", None)
    env.pop("OFA_DISABLE_PITFALLS", None)
    env["FOURC_ROOT"] = env.get("FOURC_ROOT", str(Path.home() / "4C"))
    env["FOURC_BINARY"] = env.get(
        "FOURC_BINARY", str(Path.home() / "4C/build/4C"))
    env["LD_LIBRARY_PATH"] = env.get(
        "LD_LIBRARY_PATH", "/opt/4C-dependencies/lib")
    env["PYTHONPATH"] = str(REPO / "src")
    # THE TOOLS MUST WRITE INTO THIS CELL'S SANDBOX.
    #
    # run_simulation and friends wrote to <repo>/simulation_outputs, a single
    # directory shared by every caller. An agent that followed the documented
    # OASiS workflow therefore produced its solution files where nothing
    # downstream looks, and where another cell could overwrite or read them.
    # It is the OASiS-arm tools that do this, so the cost fell entirely on the
    # arm under test: 8 of 14 coupled OASiS runs in round 1 went through it.
    if workdir is not None:
        env["OASIS_OUTPUT_DIR"] = str(Path(workdir) / "simulation_outputs")
        env["OASIS_COUPLING_DIR"] = str(Path(workdir) / "coupling")
        env["OASIS_MESH_DIR"] = str(Path(workdir) / "meshes")
        env["OASIS_BENCHMARK_DIR"] = str(Path(workdir) / "benchmark_results")

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
    client = MultiServerMCPClient({
        "oasis": {
            "command": _server_py,
            "args": ["-m", "server"],
            "cwd": str(REPO / "src"),
            "env": env,
            "transport": "stdio",
        }
    })
    return asyncio.run(client.get_tools())


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
          "the numbers you already have is graded on those numbers, and a "
          "partial one is graded on the part you supply. A verification "
          "finding — a flux imbalance, a failed conservation check — is NOT a "
          "reason to withhold a field your solve already produced: those are "
          "different verdicts and only one of them is worth zero. Write the "
          "deliverables the task asks for from what you have, state the "
          "finding alongside them, and keep working on the finding with "
          "whatever time is left."
    )


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
    tools.append(_bash_tool_for(workdir))
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


def build_mcp_agent(*, size: str, seed: int, workdir: Path, depth: int = 0):
    mcp_tools = _load_oasis_mcp_tools(workdir)
    host = _host_tools(workdir, size=size, seed=seed,
                       parent_tools=mcp_tools, depth=depth,
                       audit_on_submit=True)
    llm = _llm(size, temperature=0.2, seed=seed)
    return create_react_agent(llm, tools=mcp_tools + host,
                              prompt=MCP_SYSTEM)


__all__ = ["build_bare_agent", "build_mcp_agent"]
