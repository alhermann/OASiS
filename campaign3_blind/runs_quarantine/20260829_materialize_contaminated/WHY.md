# Void: OASiS handed these runs a working solver

C9, 27B, MCP arm, seeds 21/22/23, run 2026-08-29 as the first per-cell
iteration after Option B.

Option B stopped the coupling KNOWLEDGE from serving a finite element solve.
It did not stop `materialize_participant`, an MCP tool that shutil-copied the
COMPLETE participant file — solve included — into the agent's work_dir. The
automatic version of that delivery had already been reverted in 64a922af; the
manual one was left standing, so the same thing kept happening by a different
door.

Measured in these three runs: the tool was called 2, 2 and 4 times; 10, 8 and 8
participant files landed in the workspaces; on seeds 21 and 22 two per run
carried the full solve.

The result they produced — 3 of 3 submitting, 0 giving up, 3 of 3 reaching the
coupling tool and writing level files, against a give-up at seed 13 — is
therefore not evidence about the structural gate or about anything else. It is
kept for the record and excluded from every count.

The tool is removed and `tests/test_no_tool_installs_a_solver.py` fails on any
MCP tool that copies a participant into the agent's workspace.
