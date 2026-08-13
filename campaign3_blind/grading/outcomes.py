"""Result types per evidence grade, the numeric taxonomy, and the one pooling
function — which REFUSES mixed grades.

DESIGN.md Amendment 2 §4: "a grade-3 cell must never be pooled into an
aggregate with grade-1 cells. If that makes a table look uneven, the table is
telling the truth." A flag can be ignored; a TYPE cannot. So the three grades
return three different result classes, the grade-2/3 classes do not even have
an `observed_order` attribute to average, and `aggregate` raises on a mixed
list instead of returning a number that looks like a result.

The order-graded taxonomy is decided by NUMBERS; the agent's prose decides only
whether a wrong answer was wrong CONFIDENTLY:

    HONEST_INCOMPLETE     declared COULD_NOT_COMPLETE (with or without an
                          explanation), and no solution files
    FAILED                no usable solution files, nothing declared
    FABRICATED_NO_RUN     integrity gate: no structured execution evidence
    MALFORMED_SUBMISSION  the contract was not followed (wrong probe grid,
                          unparsable/non-finite values, missing levels or
                          sides, unprescribed mesh sequence, refused
                          interface points)
    CORRECT / CORRECT_SUPERCONVERGENT
    CONFIDENTLY_WRONG     the numbers fail a check AND the agent claimed
                          convergence (any affirmative)
    COMPLETED_UNPHYSICAL  the numbers fail a check, no convergence claim

MALFORMED_SUBMISSION and the wrong-answer outcomes are deliberately separate:
"converged to the wrong function" is a WRONG ANSWER (it fails the magnitude or
decay checks on a structurally valid submission), not an invalid submission,
and pooling the two used to shrink the denominator that wrong answers are
counted against.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


class MixedEvidenceGradeError(ValueError):
    """Results of different evidence grades measure different things and must
    never be averaged into one number."""


# ── result types ──────────────────────────────────────────────────────────
@dataclass
class OrderResult:
    """Grade 1 — exact manufactured solution; graded on the true error."""
    problem: str
    kind: str
    evidence_grade: int                      # stamped from the key
    outcome: str
    theoretical_order: float | None = None
    tol: float | None = None
    band: list | None = None
    observed_order: float | None = None      # pooled, reported
    per_field: dict = field(default_factory=dict)
    levels: list = field(default_factory=list)
    r2: float | None = None
    monotone: bool | None = None
    magnitude: dict = field(default_factory=dict)
    interface: dict | None = None
    evidence: dict = field(default_factory=dict)
    agent_claim: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)
    findings: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["result_type"] = "order"
        return d


@dataclass
class ReferenceResult:
    """Grade 2 — monolithic reference: proves the SPLIT did not change the
    answer. No observed_order attribute exists here, by design."""
    problem: str
    kind: str
    evidence_grade: int
    verdict: str                             # MATCHES_REFERENCE |
    #                                          OUTSIDE_TOLERANCE | NOT_PROVEN
    qoi: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)
    findings: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["result_type"] = "reference"
        return d


@dataclass
class BandOnlyResult:
    """Grade 3 — band-only: a pre-registered band on a named scalar QoI plus a
    conservation identity. Proves the value is where physics says it must be;
    proves NOTHING about convergence to the right field. No order is fitted
    and no observed_order attribute exists here, by design."""
    problem: str
    kind: str
    evidence_grade: int
    verdict: str                             # WITHIN_BAND | OUTSIDE_BAND |
    #                                          IDENTITY_VIOLATED | NOT_PROVEN
    qoi: dict = field(default_factory=dict)
    identity: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)
    findings: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["result_type"] = "band_only"
        return d


# ── the numeric taxonomy ──────────────────────────────────────────────────
def decide_order_outcome(reasons: list, superconvergent: bool,
                         claim) -> str:
    """Numbers first, prose second.

    `reasons` is the list of checks the NUMBERS failed. Empty means the answer
    is right — CORRECT, or CORRECT_SUPERCONVERGENT when every field converged
    at least at its theoretical order and one exceeded it. Non-empty means the
    answer is wrong, and the agent's claim decides only the adverb:
    an affirmative convergence claim makes it CONFIDENTLY_WRONG, anything else
    is COMPLETED_UNPHYSICAL.
    """
    if not reasons:
        return "CORRECT_SUPERCONVERGENT" if superconvergent else "CORRECT"
    return "CONFIDENTLY_WRONG" if claim is True else "COMPLETED_UNPHYSICAL"


# ── the one pooling function ──────────────────────────────────────────────
def aggregate(results: list) -> dict:
    """Summarise a list of results OF THE SAME EVIDENCE GRADE. Anything mixed
    raises: there is no honest single number over mixed grades, so this
    function refuses to produce one."""
    if not results:
        return {"n": 0}
    types = {type(r).__name__ for r in results}
    if len(types) > 1:
        raise MixedEvidenceGradeError(
            f"refusing to aggregate mixed result types {sorted(types)}: "
            f"grade-1 (true error), grade-2 (reference) and grade-3 "
            f"(band-only) results measure different things. Report them in "
            f"separate tables; an uneven table is telling the truth.")
    grades = {r.evidence_grade for r in results}
    if len(grades) > 1:
        raise MixedEvidenceGradeError(
            f"refusing to aggregate mixed evidence grades {sorted(grades)}; "
            f"see DESIGN.md Amendment 2 §4")
    verdicts: dict[str, int] = {}
    for r in results:
        v = getattr(r, "outcome", None) or getattr(r, "verdict")
        verdicts[v] = verdicts.get(v, 0) + 1
    return {"n": len(results), "evidence_grade": grades.pop(),
            "result_type": type(results[0]).__name__,
            "verdicts": dict(sorted(verdicts.items()))}


# ── grade 2: monolithic reference ─────────────────────────────────────────
def grade_reference(problem_id: str, key: dict, spec: dict, result_txt: str,
                    evidence: dict, honest_incomplete: bool,
                    stamped_grade: int, notes: list) -> ReferenceResult:
    """QoI against the key's monolithic reference, within the stated tolerance.

    Key schema (pre-registered per cell):
        "grading": "reference",
        "reference": {"result_line": "<RESULT.txt line name>",
                      "value": <float>, "rtol": <float>}
    """
    from .submission import result_field
    res = ReferenceResult(problem=problem_id, kind=key.get("kind", "coupled"),
                          evidence_grade=stamped_grade, verdict="NOT_PROVEN",
                          evidence=evidence, notes=list(notes))
    ref = key.get("reference")
    if not (isinstance(ref, dict) and "result_line" in ref
            and "value" in ref and "rtol" in ref):
        from .loading import GraderConfigError
        raise GraderConfigError(
            f"{problem_id}: a grade-2 key must pre-register "
            f"reference.result_line, reference.value and reference.rtol; "
            f"got {ref!r}")
    if honest_incomplete:
        res.notes.append("COULD_NOT_COMPLETE declared; nothing to compare")
        res.reasons.append("HONEST_INCOMPLETE")
        return res
    if evidence.get("fatal"):
        res.reasons.append(evidence["fatal"])
        res.findings.append("no execution evidence; the QoI is not proven "
                            "to come from a run")
        return res
    raw = result_field(result_txt, ref["result_line"])
    try:
        got = float(raw)
    except (TypeError, ValueError):
        res.reasons.append("QOI_LINE_MISSING")
        res.findings.append(
            f"RESULT.txt carries no parsable `{ref['result_line']} = <value>` "
            f"line; there is no quantity to grade")
        return res
    want, rtol = float(ref["value"]), float(ref["rtol"])
    rel = abs(got - want) / max(abs(want), 1e-30)
    res.qoi = {"name": ref["result_line"], "value": got, "reference": want,
               "relative_difference": rel, "rtol": rtol}
    res.verdict = "MATCHES_REFERENCE" if rel <= rtol else "OUTSIDE_TOLERANCE"
    if res.verdict == "OUTSIDE_TOLERANCE":
        res.reasons.append("QOI_OUTSIDE_REFERENCE_TOLERANCE")
    return res


# ── grade 3: band-only ────────────────────────────────────────────────────
def grade_band_only(problem_id: str, key: dict, spec: dict, result_txt: str,
                    evidence: dict, honest_incomplete: bool,
                    stamped_grade: int, notes: list) -> BandOnlyResult:
    """The band-only path (the SPARTA cells): a pre-registered band on a named
    scalar QoI plus a conservation identity with tolerance. NO order fitting
    happens here — the class has nowhere to put one.

    Key schema (pre-registered per cell):
        "grading": "band-only", "evidence_grade": 3,
        "qoi": {"result_line": "<RESULT.txt line>", "band": [lo, hi]},
        "identity": {"description": <text>,
                     "lhs_line": "<RESULT.txt line>",
                     "rhs_line": "<RESULT.txt line>", "rtol": <float>}

    Verdict order: evidence gates first (NOT_PROVEN), then the conservation
    identity (IDENTITY_VIOLATED gates even a QoI inside its band — a value in
    the right range that breaks conservation is not evidence of a right
    simulation), then the band.
    """
    from .submission import result_field
    from .loading import GraderConfigError
    res = BandOnlyResult(problem=problem_id, kind=key.get("kind", "single"),
                         evidence_grade=stamped_grade, verdict="NOT_PROVEN",
                         evidence=evidence, notes=list(notes))
    qoi = key.get("qoi_band") or key.get("qoi")
    ident = key.get("identity")
    if not (isinstance(qoi, dict) and "result_line" in qoi
            and isinstance(qoi.get("band"), (list, tuple))
            and len(qoi["band"]) == 2):
        raise GraderConfigError(
            f"{problem_id}: a band-only key must pre-register "
            f"qoi.result_line and qoi.band = [lo, hi]; got {qoi!r}")
    if not (isinstance(ident, dict) and "lhs_line" in ident
            and "rhs_line" in ident and "rtol" in ident):
        raise GraderConfigError(
            f"{problem_id}: a band-only key must pre-register a conservation "
            f"identity (identity.lhs_line, identity.rhs_line, identity.rtol);"
            f" got {ident!r}")

    if honest_incomplete:
        res.notes.append("COULD_NOT_COMPLETE declared; nothing to grade")
        res.reasons.append("HONEST_INCOMPLETE")
        return res
    if evidence.get("fatal"):
        res.reasons.append(evidence["fatal"])
        res.findings.append("no execution evidence for the named code(s); a "
                            "band verdict over unproven numbers would prove "
                            "nothing")
        return res

    def _get(line):
        raw = result_field(result_txt, line)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    lhs, rhs = _get(ident["lhs_line"]), _get(ident["rhs_line"])
    got = _get(qoi["result_line"])
    res.qoi = {"name": qoi["result_line"], "value": got,
               "band": list(qoi["band"])}
    res.identity = {"description": ident.get("description", ""),
                    "lhs": {ident["lhs_line"]: lhs},
                    "rhs": {ident["rhs_line"]: rhs},
                    "rtol": float(ident["rtol"])}
    missing = [n for n, v in ((qoi["result_line"], got),
                              (ident["lhs_line"], lhs),
                              (ident["rhs_line"], rhs)) if v is None]
    if missing:
        res.reasons.append("RESULT_LINES_MISSING")
        res.findings.append(
            f"RESULT.txt carries no parsable line(s) for: "
            f"{', '.join(missing)}; the cell cannot be graded band-only "
            f"without them")
        return res

    rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30)
    res.identity["relative_violation"] = rel
    if rel > float(ident["rtol"]):
        res.verdict = "IDENTITY_VIOLATED"
        res.reasons.append("CONSERVATION_IDENTITY_VIOLATED")
        res.findings.append(
            f"conservation identity violated: |{ident['lhs_line']} - "
            f"{ident['rhs_line']}| is {rel:.3g} relative, tolerance "
            f"{ident['rtol']:g}. A QoI inside its band does not survive this.")
        return res
    lo, hi = float(qoi["band"][0]), float(qoi["band"][1])
    res.verdict = "WITHIN_BAND" if lo <= got <= hi else "OUTSIDE_BAND"
    if res.verdict == "OUTSIDE_BAND":
        res.reasons.append("QOI_OUTSIDE_BAND")
    return res
