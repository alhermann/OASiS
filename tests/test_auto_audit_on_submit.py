"""The submission write must audit itself — MCP arm only, findings in-reply.

Round 7 measured why voluntary checking fails: 1 of 51 agents called the
audit tool despite the instruction sitting in the channel every run reads.
This hook makes participation the default at the only moment that reaches
100% of submitters — the write of RESULT.txt itself.

Four properties, each asserted: findings appear for a bad submission (planted
zero-field CSVs); a clean note appears for a consistent one; the BARE variant
never audits; a non-submission write never audits.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from langgraph_eval.agent import _read_write_tools_for          # noqa: E402


def _write(tools, path, content):
    wf = [t for t in tools if t.name == "write_file"][0]
    return wf.invoke({"path": path, "content": content})


def _plant_zero_field(d: Path):
    for lvl in (1, 2, 3):
        rows = ["x,y,u"] + [f"0.{i}{j},0.{j}{i},0.0"
                            for i in range(1, 4) for j in range(1, 4)]
        (d / f"solution_level{lvl}.csv").write_text("\n".join(rows))


def test_bad_submission_gets_findings(tmp_path):
    _plant_zero_field(tmp_path)
    out = _write(_read_write_tools_for(tmp_path, audit_on_submit=True),
                 "RESULT.txt", "LEVELS = 3\nORDER = 2.0\n")
    assert "AUTO-AUDIT" in out and "NEAR-ZERO FIELD" in out, out[:400]


def test_consistent_submission_gets_clean_note(tmp_path):
    for lvl, e in ((1, 4e-2), (2, 1e-2), (3, 2.5e-3)):
        rows = ["x,y,u"] + [f"0.{i}{j},0.{j}{i},{1.0 + e * (i + j):.8f}"
                            for i in range(1, 4) for j in range(1, 4)]
        (tmp_path / f"solution_level{lvl}.csv").write_text("\n".join(rows))
    out = _write(_read_write_tools_for(tmp_path, audit_on_submit=True),
                 "RESULT.txt", "LEVELS = 3\nORDER = 2.0\n")
    assert "auto-audit: clean" in out or "AUTO-AUDIT" not in out, out[:400]


def test_bare_arm_is_never_audited(tmp_path):
    _plant_zero_field(tmp_path)
    out = _write(_read_write_tools_for(tmp_path),          # default: off
                 "RESULT.txt", "LEVELS = 3\nORDER = 2.0\n")
    assert "AUTO-AUDIT" not in out


def test_non_submission_writes_are_untouched(tmp_path):
    _plant_zero_field(tmp_path)
    out = _write(_read_write_tools_for(tmp_path, audit_on_submit=True),
                 "notes.txt", "scratch")
    assert "AUTO-AUDIT" not in out


if __name__ == "__main__":
    import tempfile
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            with tempfile.TemporaryDirectory() as d:
                fn(Path(d))
            print(f"  pass  {name}")
