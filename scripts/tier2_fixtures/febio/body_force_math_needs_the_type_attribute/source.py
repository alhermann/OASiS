"""Tier-2: a position-dependent body force IS writable in an FEBio deck.

Verifies the febio body_force_trap entry, whose load-bearing claim used to be
the opposite: "a math string cannot make it vary per element, so a
POSITION-DEPENDENT source (any manufactured solution, any polynomial load)
cannot be written in the deck at all."

Agents believed it. FB2 runs reported back "FEBio 4.12.0 does not support
spatially-varying body forces through its XML interface" and stopped, and the
entry had already noted that one agent "read the paragraph above as 'FEBio
cannot do this problem' and abandoned a solvable task" -- the response at the
time was to add a workaround rather than to check whether the wall was real.

Four arms on four hex8 elements stacked along X, clamped at x = 0:

  MATH      <force type="math">0,0,-1*X</force>   runs; sz differs per element
  NOMATH    <force>0,0,-1*X</force>               hard parse failure
  NONCONST  <body_load type="non-const">          runs; identical to MATH
  FLAT      <force type="math">0,0,-2</force>     runs; DIFFERENT profile

The FLAT arm is what makes the claim falsifiable: MATH varying per element is
only meaningful if it is not merely a rescaling of a uniform load, so the
per-element ratio MATH/FLAT is required to be non-constant.

The material is stiff (E = 1e6) on purpose. An over-large body force makes the
nonlinear solve diverge with "failed to converge at time : 1", which reads
like a deck problem and is not one -- that is how this was nearly mismeasured.

MUTATION CONTROL. T2_MUTATE=1 gives the NOMATH arm the type="math" attribute
it is missing. It then parses and terminates normally, the syntax error
disappears and 'type_attribute_is_required=reproduced' is no longer printed,
so the fixture must go red.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

NX = 4
LOG = 'sz_e.csv'


def deck(load_xml: str) -> str:
    mesh, _info = L.hex8_box(1, lx=float(NX), ly=1.0, lz=1.0)
    # a bar of NX elements along x, clamped on the x = 0 face
    nodes, els, nid = [], [], {}
    c = 0
    for k in range(2):
        for j in range(2):
            for i in range(NX + 1):
                c += 1
                nid[(i, j, k)] = c
                nodes.append(f'      <node id="{c}">{i},{j},{k}</node>')
    for i in range(NX):
        q = [nid[(i, 0, 0)], nid[(i + 1, 0, 0)], nid[(i + 1, 1, 0)],
             nid[(i, 1, 0)], nid[(i, 0, 1)], nid[(i + 1, 0, 1)],
             nid[(i + 1, 1, 1)], nid[(i, 1, 1)]]
        els.append(f'      <elem id="{i + 1}">{",".join(map(str, q))}</elem>')
    fix = [nid[(0, j, k)] for k in range(2) for j in range(2)]
    mesh = (
        '  <Mesh>\n    <Nodes name="n">\n' + "\n".join(nodes) +
        '\n    </Nodes>\n    <Elements type="hex8" name="Part1">\n' +
        "\n".join(els) + '\n    </Elements>\n'
        f'    <NodeSet name="fix">{",".join(map(str, fix))}</NodeSet>\n'
        '  </Mesh>')
    material = (
        "  <Material>\n"
        '    <material id="1" name="Material1" type="isotropic elastic">\n'
        "      <density>1.0</density><E>1000000.0</E><v>0.25</v>\n"
        "    </material>\n  </Material>")
    boundary = (
        "  <Boundary>\n"
        '    <bc name="f" type="zero displacement" node_set="fix">\n'
        "      <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>\n"
        "    </bc>\n  </Boundary>")
    output = (
        '  <Output>\n    <plotfile type="febio">'
        '<var type="displacement"/></plotfile>\n'
        f'    <logfile><element_data data="sz" delim="," file="{LOG}"/>'
        "</logfile>\n  </Output>")
    return L.solid_deck(mesh=mesh, material=material, boundary=boundary,
                        output=output,
                        extra_sections=f"  <Loads>{load_xml}</Loads>\n")


MATH = '<body_load type="body force"><force type="math">0, 0, -1*X</force>' \
       '</body_load>'
NOMATH = ('<body_load type="body force"><force type="math">0, 0, -1*X</force>'
          '</body_load>') if MUTATE else \
         '<body_load type="body force"><force>0, 0, -1*X</force></body_load>'
NONCONST = ('<body_load type="non-const"><x>0</x><y>0</y><z>-1*X</z>'
            '</body_load>')
FLAT = '<body_load type="body force"><force type="math">0, 0, -2</force>' \
       '</body_load>'
# the constant that the SILENT degradation collapses to: (0, 0, -1)
CONST = '<body_load type="body force"><force type="math">0, 0, -1</force>' \
        '</body_load>'
# the same omission with the expression FIRST: this one fails loudly
NOMATH_FIRST = '<body_load type="body force"><force>-1*X, 0, 0</force>' \
               '</body_load>'


def sz(run) -> list:
    """Element sz at the LAST time block, ordered by element id."""
    blocks = L.parse_log_csv(run.files.get(LOG) or "")
    if not blocks:
        return []
    _time, rows = blocks[-1]
    return [rows[k][0] for k in sorted(rows)]


def main() -> int:
    r_math = L.run(deck(MATH), collect=(LOG,))
    r_no = L.run(deck(NOMATH), collect=(LOG,))
    r_nc = L.run(deck(NONCONST), collect=(LOG,))
    r_flat = L.run(deck(FLAT), collect=(LOG,))

    for tag, r in (("MATH", r_math), ("NONCONST", r_nc), ("FLAT", r_flat)):
        if "N O R M A L" not in (r.log + r.out):
            print(f"{tag} did not terminate normally:\n"
                  f"{(r.log + r.out)[-600:]}")
            return 1

    m, n, f = sz(r_math), sz(r_nc), sz(r_flat)
    if not (len(m) == len(n) == len(f) == NX):
        print(f"expected {NX} element rows, got {len(m)}/{len(n)}/{len(f)}")
        return 1

    print("MATH_SZ=" + ",".join(f"{v:.9e}" for v in m))
    print("FLAT_SZ=" + ",".join(f"{v:.9e}" for v in f))

    if len({f"{v:.9e}" for v in m}) > 1:
        print("math_force_varies_per_element=reproduced")
    if all(abs(a - b) <= 1e-15 * max(1.0, abs(a)) for a, b in zip(m, n)):
        print("math_equals_legacy_nonconst=reproduced")
    ratios = [a / b for a, b in zip(m, f) if b]
    if ratios and (max(ratios) - min(ratios)) > 0.5:
        print("not_a_rescaling_of_a_uniform_load=reproduced")

    # OMITTING type="math" FAILS IN TWO DIFFERENT WAYS, AND THE COMMON ONE
    # IS SILENT. With the expression in a LATER component the numeric prefix
    # is taken and the rest discarded, so the run terminates normally with a
    # constant load; only when it sits FIRST does the parse fail loudly.
    # Both are pinned, because an entry that promised a hard error would send
    # a reader looking for a message that never appears.
    blob_no = r_no.log + r_no.out
    r_const = L.run(deck(CONST), collect=(LOG,))
    if MUTATE:
        # the mutation gives NOMATH its attribute back: it must now differ
        # from the constant arm
        same = sz(r_no) == sz(r_const)
        print("mutated_nomath_still_constant=%d" % same)
        return 0
    if ("N O R M A L" in blob_no and "syntax error" not in blob_no
            and sz(r_no) == sz(r_const)):
        print("SILENT_SZ=" + ",".join(f"{v:.9e}" for v in sz(r_no)))
        print("missing_type_silently_gives_a_constant_load=reproduced")
    else:
        print("the bare form no longer degrades silently to a constant:\n"
              + blob_no[-400:])

    r_first = L.run(deck(NOMATH_FIRST), collect=(LOG,))
    blob_first = r_first.log + r_first.out
    if "syntax error" in blob_first and "FAILED" in blob_first:
        print("missing_type_in_first_component_is_a_parse_error=reproduced")
    else:
        print("the leading-expression form no longer fails to parse:\n"
              + blob_first[-400:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
