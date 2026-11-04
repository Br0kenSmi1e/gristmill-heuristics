At session start, read tools/joint_search/LOOP.md, then BUGS.md, then
CLASSES.md, then NOTES.md in this repository (a worktree on branch
joint-search-proto). Those files are authoritative: they define the
workflow, the current verified state, and the work queue.

This arm develops a joint ILP+DP selector (a challenger for the
mainline's contested factorization slot). The terminal goal is stated in
LOOP.md ("The goal"): on every train problem, an emitted candidate that
verifies AND strictly beats the same-equation gristmill FLOP count at
(16,64) — that clause is the space-completeness test. Work autonomously — pull the
next item from BUGS.md (bugs, currently B-006) or CLASSES.md
(expressiveness classes), run experiments yourself, commit validated
work as you go.

Key mechanics:
- Python environment: tools/joint_search/.venv (numpy+scipy; recreate
  with `uv venv tools/joint_search/.venv && uv pip install numpy scipy`).
- The harness is prebuilt at target/debug/harness; if stale, rebuild
  with `cargo build` on a quiet machine (check uptime first).
- Gates per LOOP.md: microtests.py, emission self-check (ILP objective
  vs emitted price), harness check (truthmill validate + numerical
  oracle at (3,5)), budget check. Log via run_battery.py.
- Verified numbers only: a count counts only after truthmill::verify +
  the numerical oracle pass. Cost-space ILP optima are labeled
  relaxations and never enter the verified scoreboard.
- The mainline repo at /home/longli/code/gristmill-heuristics is
  READ-ONLY context (its PROTOCOL.md/RESEARCH.md explain the wider
  project; its benches/ are the inputs). Never edit it, never commit
  there, never run its timing commands — another agent works there.
  All work happens in this worktree, committed to joint-search-proto.
- Protected problems (never tuned against): ccsd_t, eom_ccsd,
  adc2-heldout, ccsd_lambda. ccsd_t execution is a hard memory blocker
  (F-017): do not run it full-size.

Stop only at a resource limit, a concrete blocker, or a milestone
boundary. Report findings; do not ask permission for ordinary
investigative choices. If this prompt and the repo files disagree, the
files win — report the conflict.
