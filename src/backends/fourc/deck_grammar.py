"""The 4C deck grammar — ONE copy, served by every path that needs it.

WHY THIS MODULE EXISTS. 4C is a BINARY that consumes a YAML deck, so the deck is
this backend's run interface in exactly the way exports.json is the coupling
driver's: it cannot be guessed, and no solve happens without it. OASiS elides
"the solve" by design, which is right for a Python library and wrong here.

MEASURED. The coupling payload for solver='fourc' contained no PROBLEM TYPE, no
MATERIALS, no SCALAR TRANSPORT DYNAMIC, no DESIGN LINE DIRICH, no NODE COORDS and
no NUMDOF; all three OASiS-arm runs of coupled cell C2 died on the 4C side, and
seed71 named the cause itself ("4C scalar transport module requires specific
topology definitions"). The SINGLE-CODE 4C cells (FC1, FC2) were worse off
still: they never call knowledge(topic='coupling'), so they received none of it.

AND IT MUST NOT BECOME FOUR COPIES. The interface-probe text taught that lesson
already: it existed in four places, two of them dead, and a fix applied to the
wrong one looked correct for hours. So the grammar lives here once and both
serving paths import it.

WHAT THIS IS NOT. It is a GRAMMAR, not a solve: every number is an arbitrary
placeholder, and the agent still derives its own mesh, materials, boundary data
and source term. Verified by dogfooding — a deck written from this text alone
runs C2 side A to completion (tests/fixtures/fourc_running_deck/).
"""

FOURC_DECK_GRAMMAR = """\
THE 4C DECK GRAMMAR — you cannot guess it, and 4C is a BINARY, so the deck is
this backend's run interface in exactly the way exports.json is the driver's.
Every NUMBER below is an ARBITRARY PLACEHOLDER; the sections and their syntax
are what is documented. A deck with these sections runs; one missing any of
them aborts during input parsing.

  TITLE:
    - "anything"
  PROBLEM SIZE:
    ELEMENTS: 4
    NODES: 9
  PROBLEM TYPE:
    PROBLEMTYPE: "Scalar_Transport"
  SCALAR TRANSPORT DYNAMIC:
    TIMEINTEGR: "Stationary"
    SOLVERTYPE: "linear_full"
    NUMSTEP: 1
    TIMESTEP: 1.0
    MAXTIME: 1.0
    LINEAR_SOLVER: 1
  SOLVER 1:
    SOLVER: "UMFPACK"
  MATERIALS:
    - MAT: 1
      MAT_scatra:
        DIFFUSIVITY: 1.0          # <- YOUR k
  FUNCT1:
    - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "x^2*y"     # <- YOUR source/BC
  DESIGN LINE DIRICH CONDITIONS:
    - E: 1
      NUMDOF: 1
      ONOFF: [1]
      VAL: [0.0]
      FUNCT: [0]
  NODE COORDS:
    - "NODE 1 COORD 0.0 0.0 0.0"
  TRANSPORT ELEMENTS:
    - "1 TRANSP QUAD4 1 2 5 4 MAT 1 TYPE Std"
  DLINE-NODE TOPOLOGY:
    - "NODE 1 DLINE 1"

THE VOLUME SOURCE TERM f IS A "SURF" NEUMANN CONDITION IN 2-D. There is no
body-force section: 4C calls a 2-D domain a SURFACE, so `-div(k grad u) = f`
gets its f from

  DESIGN SURF TRANSPORT NEUMANN CONDITIONS:
    - E: 1
      NUMDOF: 1
      ONOFF: [1]
      VAL: [1.0]        # scale; the shape comes from FUNCT
      FUNCT: [1]        # -> FUNCT1's SYMBOLIC_FUNCTION_OF_SPACE_TIME
  DSURF-NODE TOPOLOGY:
    - "NODE 1 DSURFACE 1"      # every node of the subdomain

In 3-D the same role is played by `DESIGN VOL TRANSPORT NEUMANN CONDITIONS`
with `DVOL-NODE TOPOLOGY`. Using the VOL form on a 2-D problem is trap (d)
below. Omit this section and you solve f = 0 -- the run succeeds and the
answer is wrong, which is the worst failure mode available.

FOUR MEASURED WAYS THIS DIES, all of them silently:
(a) THE LEGACY BLOCKS ARE YAML SEQUENCES. `NODE COORDS`, `TRANSPORT
    ELEMENTS` and `D*-NODE TOPOLOGY` entries each need `- ` and quotes.
    Written bare, YAML reads them as mapping keys and 4C dies with
      ERROR: could not find ':' colon after key
    BEFORE its own banner appears.
(b) `**` IS NOT EXPONENTIATION in SYMBOLIC_FUNCTION_OF_SPACE_TIME. Use `^`.
    The task states its source term in Python notation; rewrite every term.
(c) ONOFF / VAL / FUNCT must each have EXACTLY `NUMDOF` entries, or
      [!] Candidate parameter 'VAL' has incorrect size
(d) A condition's dimension must not exceed the problem's: a
    `DESIGN VOL ...` block on a 2-D problem gives
      Dimension of condition is larger than the problem dimension.

AND READ THE LOG FROM THE TOP. 4C buffers stdout and MPI_Abort kills it
before the flush, so `| tail` shows only MPI boilerplate. Measured on one
failing deck: 8 lines without line buffering, 43 with it. Run
    stdbuf -oL <4C binary> deck.4C.yaml out > run.log 2>&1 ; head -40 run.log
`Invalid MIT-MAGIC-COOKIE-1 key` is an X11 warning that appears on
SUCCESSFUL runs too — `4C -p` prints it and then dumps the whole grammar.
It never explains a failure.
"""
