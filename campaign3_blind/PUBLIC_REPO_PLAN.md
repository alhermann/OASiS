# Curating the public GitHub repo (a product, not a mirror)

Alexander, 2026-08-15: the public repo is a product for the community — only
what a user can use; the README is the front door and must be a real
get-started manual; no tests, dev scaffolding, private paths, or campaign
data. This file records what is actually in the way. It is itself a
development note and must NOT ship.

## Measured state (2026-08-15)

Good: the blind campaign has NEVER been pushed — `campaign3_blind/` is 0
files on `hereon/main`. The evaluation keys, runs, grades and findings are
private and stay that way.

Already public and questionable under the new rule:

| path | public files | verdict |
|---|---|---|
| `scripts/` | 280 | mostly `tier2_fixtures/` — our private verification corpus |
| `tests/` | 103 | internal test scaffolding |
| `langgraph_eval/` | 5 | the LLM evaluation harness for the paper, not a user feature |
| `benchmarks/` | 41 | mixed: coupling examples ARE useful to users; ledgers are not |
| `data/` | 29 | mixed: catalogs are product; measurement records are not |
| `src/` | 227 | the product |

## The one hard product bug

`src/backends/_installed_api.py` ships THIS MACHINE's absolute paths as
served knowledge — e.g. a "run" line telling any user to invoke
`/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python`, a deal.II
build tree at `/home/alexander/dealii/build`, and 4C at
`/home/alexander/4C/build/4C`. Eight files under `src/` contain
`/home/alexander`. For a community user this is both useless and revealing.

The right fix is NOT to hand-edit the paths: this file is a record of what is
installed HERE, and the campaign's MCP arm depends on it being accurate. The
product fix is to generate it per installation from autodiscovery and ship a
path-free template, with the local record kept out of the published tree.

DO NOT change it while the campaign is running: `src/backends/**` is now part
of the pre-registration and is the MCP arm's treatment. Changing it is a
knowledge change that invalidates cross-round comparisons. Schedule it for
after the campaign, or make the change and re-run the affected cells.

## Plan for the public release (do NOT execute mid-campaign)

1. Build the public tree deliberately — a curated branch, not a merge of the
   development branch. Deleting files in a later commit does not remove them
   from history; the public branch should never receive them in the first
   place.
2. Ship: `src/` (path-free), a small set of runnable examples, `README.md`,
   `LICENSE`, `CITATION.cff`, `CONTRIBUTING.md`, `pyproject.toml`, `logo/`,
   and a lean smoke-test suite a user can run to check their install.
3. Do not ship: `campaign3_blind/`, `langgraph_eval/`, `scripts/tier2_*`,
   measurement records under `data/`, benchmark ledgers, and every
   development `.md` (DEV_FINDINGS, CONVERGENCE, this file, audit notes).
4. Scrub absolute paths from everything that ships; add a CI check that fails
   on `/home/`, `/mnt/`, or any user directory in the published tree.
5. Rewrite `README.md` as a get-started manual: install in one block, first
   run in five lines, one worked tutorial end to end, a short table of
   backends with what each needs, then pointers. Current README is 248 lines
   and reads as a project description rather than a manual — it needs a real
   quick-start at the top.
6. Verify the result by cloning the public repo fresh into a clean directory
   and following the README as a new user, with nothing from this machine on
   the path.
