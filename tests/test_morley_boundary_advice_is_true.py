"""The Morley boundary-condition advice must be true of the installed skfem.

Why this test exists. The served biharmonic template used to carry the comment
"Simply supported: u=0 on boundary" above a call that constrains BOTH u and
du/dn — which is clamped. It survived the existing signal-verification floor
because that floor only asks whether a claim NAMES a real entity and uses
symptom vocabulary. This one did both, and was false.

The distinction is not cosmetic: under the same unit load, simple support
gives a peak deflection several times the clamped one. An agent handed the
wrong call gets a wrong answer with no error message.

So this test executes the advice instead of reading it:
  1. the two boundary conditions are separable and materially different
  2. the call served for simple support really is the value dofs
  3. .facet_ix — the neighbouring attribute — is NOT global, which is why the
     served text tells agents to address the blocks by name
  4. the dof counts stated in the comment match the mesh the template builds

Guard against the reverse failure too: if a future skfem makes facet_ix global,
assertion 3 fails loudly and the trap warning must be retired rather than left
to mislead.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    import numpy as np
    from skfem import (
        Basis,
        BilinearForm,
        ElementTriMorley,
        LinearForm,
        MeshTri,
        asm,
        condense,
        solve,
    )

    HAVE_SKFEM = True
except ImportError:  # pragma: no cover - environment without skfem
    HAVE_SKFEM = False

from src.backends.skfem.generators.biharmonic import GENERATORS

TEMPLATE = GENERATORS["biharmonic_2d"]({})


def _basis(refine_level: int):
    return Basis(MeshTri.init_symmetric().refined(refine_level), ElementTriMorley())


@unittest.skipUnless(HAVE_SKFEM, "skfem not installed in this interpreter")
class TestMorleyBoundaryAdvice(unittest.TestCase):
    def _solve(self, ib, D):
        @BilinearForm
        def biharmonic(u, v, w):
            return (
                u.hess[0][0] * v.hess[0][0]
                + u.hess[1][1] * v.hess[1][1]
                + 2 * u.hess[0][1] * v.hess[0][1]
            )

        @LinearForm
        def load(v, w):
            return 1.0 * v

        K, f = asm(biharmonic, ib), asm(load, ib)
        return solve(*condense(K, f, D=D))

    def test_the_served_simple_support_call_is_the_value_dofs(self):
        """d.nodal['u'] is what simple support means: values fixed, slope free."""
        for level in (3, 4):
            with self.subTest(level=level):
                d = _basis(level).get_dofs()
                self.assertEqual(
                    sorted(d.nodal["u"]),
                    sorted(d.nodal_ix),
                    "nodal_ix diverged from nodal['u']; the served call for "
                    "simple support no longer selects the value dofs",
                )

    def test_clamped_uses_both_blocks_and_nothing_else(self):
        """flatten() is exactly the value dofs plus the normal-derivative dofs."""
        for level in (3, 4):
            with self.subTest(level=level):
                d = _basis(level).get_dofs()
                self.assertEqual(
                    sorted(d.flatten()),
                    sorted(list(d.nodal["u"]) + list(d.facet["u_n"])),
                    "flatten() is no longer the union of the two blocks, so "
                    "the served clamped call means something else now",
                )

    def test_facet_ix_is_not_global_which_is_why_we_warn_about_it(self):
        """The trap the served text names: facet_ix indexes inside the block."""
        d = _basis(3).get_dofs()
        self.assertNotEqual(
            sorted(d.facet_ix),
            sorted(d.facet["u_n"]),
            "facet_ix is now global — the trap warning in the biharmonic "
            "template is obsolete and must be removed, not left misleading",
        )

    def test_the_two_boundary_conditions_are_materially_different(self):
        """Wrong BC is a wrong answer, not a wrong decimal."""
        ib = _basis(4)
        d = ib.get_dofs()
        clamped = self._solve(ib, d.flatten())
        supported = self._solve(ib, d.nodal["u"])
        ratio = supported.max() / clamped.max()
        self.assertGreater(
            ratio,
            2.0,
            f"peak deflection ratio {ratio:.3f} — if the two boundary "
            "conditions have converged, the emphasis in the served text "
            "should be revisited",
        )

    def test_the_counts_in_the_comment_match_the_mesh_the_template_builds(self):
        """The defect that prompted this file: numbers describing another mesh.

        The comment once quoted the refined(2) counts above a refined(4) mesh.
        Whatever counts the template states must hold for the mesh it builds.
        """
        level = int(re.search(r"\.refined\((\d+)\)", TEMPLATE).group(1))
        ib = _basis(level)
        d = ib.get_dofs()
        actual = {
            len(d.flatten()),
            len(d.nodal["u"]),
            len(d.facet["u_n"]),
            ib.N,
        }
        # Any bare integer in the boundary-condition comment block must be a
        # real count of this mesh, an index the text quotes from facet_ix, or
        # the small integers used for prose (0, 1, 2).
        block = TEMPLATE.split("# BOUNDARY CONDITIONS ON A MORLEY ELEMENT")[1]
        block = block.split("D = d.flatten()")[0]
        quoted_indices = set(d.facet_ix.tolist()) | set(d.facet["u_n"].tolist())
        for token in re.findall(r"(?<![\w.])(\d+)(?![\w.])", block):
            value = int(token)
            if value <= 2:
                continue
            self.assertTrue(
                value in actual or value in quoted_indices,
                f"the comment states {value}, which is not a count of the "
                f"refined({level}) mesh it sits above (counts are {sorted(actual)}) "
                "nor an index it quotes — this is the refined(2)-vs-refined(4) "
                "defect returning",
            )

    def test_the_template_still_runs(self):
        """Advice that does not execute is not advice (paper section 3.1)."""
        ib = _basis(4)
        u = self._solve(ib, ib.get_dofs().flatten())
        self.assertTrue(np.all(np.isfinite(u)))
        self.assertGreater(u.max(), 0.0)


if __name__ == "__main__":
    unittest.main()
