# LOOP — how the challenger arm works

> **Fresh session?** Read `LOOP.md` (this file), then `BUGS.md` (open work),
> then `CLASSES.md` (what's been tried, with measured values), then
> `NOTES.md` (history). Run the battery to see the current state:
> `/tmp/proto-venv/bin/python tools/joint_search/run_battery.py "<note"`
> from the worktree root. The first open work item is B-006.
> The repo this branch lives in is shared with another agent — never edit
> mainline files; never run timing commands except at milestones.

This directory is a self-contained research loop developing the joint
ILP+DP selector (a challenger for the mainline's contested factorization
slot). It follows the mainline's PROTOCOL.md in miniature, adapted to
what this arm actually measures: **verified FLOP ratios vs the
same-equation gristmill baseline**.

Theoretical background: the structure-tensor note
(`~/Documents/tensor-factorization-encoding.pdf`) and
`docs/joint-search-design.md` (literature check + plan). Current state:
`NOTES.md` (running log) and `SCOREBOARD.md` (verified numbers).

## The fast loop (one iteration)

One iteration = ONE entry from BUGS.md or CLASSES.md, carried through its
gates. Never "improve the scoreboard" in the abstract.

1. SELECT — highest-priority open B-entry, or next C-entry by measured
   value.
2. REPRODUCE — pin a minimal fixture FIRST (term-subset bisection via
   `bisect.py`; convention disputes via `microtests.py`). No fixture, no
   fix.
3. DIAGNOSE — verify the mechanism and record it in the entry BEFORE the
   fix. Mechanism chooses the fix type.
4. FIX — one targeted change.
5. GATE, cheapest first, all must pass:
   a. `python3 microtests.py` (conventions; seconds)
   b. emission self-check: ILP objective vs emitted price agree within
      lincomb accounting (structural divergence = stop)
   c. `harness check` (truthmill validate + numerical oracle at (3,5))
      on every touched problem AND the full train battery
   d. budget: ILP build+solve stays < 10 s per problem at 63 terms
   e. benchmark spot-checks ONLY at milestones, ONLY under the load guard
6. LOG — one JSON line in `trials/` via `run_battery.py` (never edited
   by hand).
7. COMPRESS — every ~10 iterations: fold closed entries into NOTES.md
   invariants; refactor if fix patterns reveal structural debt (two
   emission rewrites so far came from patch-on-patch drift — prefer
   rewriting a function over fifth-order patching).

## Standing rules

- **Verified numbers only.** A number counts when the candidate passes
  truthmill::verify + the numerical oracle. Cost-space ILP optima are
  labeled as such and never mixed into the scoreboard.
- **Same-equation baselines only.** Comparisons use
  `benches/candidates/*_gristmill.json` (the mainline's exports).
- **Load guard.** No timing commands in this arm except at milestones;
  the mainline agent shares the machine. `run_battery.py` records load
  at start and refuses above 8.
- **Protected sets** (inherited from mainline): never tune on
  {ccsd_t, eom_ccsd, adc2-heldout, ccsd_lambda}. This arm develops on
  {mp3, ccd, ccsd, adc2-train} only.
- **Symmetry signs.** Any sym-quotiented solve must be emitted with
  sign-correct folding and pass the oracle before its number counts.
  Cost-space sym numbers are relaxations — label them.
- **Conventions are pinned by microtests, not re-derived.** If a
  convention question arises (permute direction, orbit signs, ...),
  settle it with a numpy microtest and add it to `microtests.py`.

## Milestones (event-driven)

Fire when: a C-entry lands on the verified scoreboard; a B-entry class
closes; or 10–15 iterations pass. A milestone = run the full battery +
write SCOREBOARD.md + critic pass (fresh-context review of the entries
and trial lines; file challenges as new B-entries).
