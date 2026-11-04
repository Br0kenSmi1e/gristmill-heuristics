# SCOREBOARD — verified numbers (challenger arm)

Source: trials/20261104-190100.jsonl (full battery, all gates green).
All candidates emitted by the global arm, verified by truthmill::verify +
numerical oracle at (3,5); FLOP ratios at (16,64) vs the same-equation
gristmill baselines (benches/candidates/*_gristmill.json).

## Verified scoreboard (sym ON unless noted)

| problem | ratio | goal (<1.0) | gap lever |
|---|---|---|---|
| mp3   | 0.6566 | PASS | — |
| ccd   | 1.0163 | fail | realization gap vs 0.878 relaxed: C-010 lock, sym depth |
| ccsd  | 1.1039 | fail | realization gap vs 0.766 relaxed: C-010 lock |
| adc2  | 1.0031 | fail | B-007 lincomb accounting (~7.3M over objective 0.9925x) |

No-sym (fully legal space, verified): ccd 1.3895, ccsd 1.5055 — sym
exploitation is the lever on both.

## Cost-space relaxations (NOT verified; upper bounds on the space)

sym-relaxed ILP optima: mp3 0.657, ccd 0.878, ccsd 0.766, adc2 0.748.
These predate the C-010 soundness lock (perm-membered reps excluded from
partial-keep families); the emitted solves above include the lock.

## Milestone history

- 20261104-190100: B-006 closed (formation orbit signs + perm-member
  double-emit). First all-green battery with sym on all four problems.
- 20261104-181011: pre-B-006 state (ccsd verify failing).
