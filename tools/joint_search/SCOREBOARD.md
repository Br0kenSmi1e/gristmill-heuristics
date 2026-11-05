# SCOREBOARD — verified numbers (challenger arm)

Source: trials/20261104-201700.jsonl (full battery, all gates green).
All candidates emitted by the global arm, verified by truthmill::verify +
numerical oracle at (3,5); FLOP ratios at (16,64) vs the same-equation
gristmill baselines (benches/candidates/*_gristmill.json).

## Verified scoreboard (sym ON; adc2/mp3 have no declared symmetries)

| problem | ratio | goal (<1.0 strict) |
|---|---|---|
| mp3   | 0.6564 | PASS |
| ccd   | 0.9919 | PASS |
| ccsd  | 0.9348 | PASS |
| adc2  | 1.000000 | EXACT TIE — C-014 open blocker |

ILP objective == emitted price − const on all four (residual ≤ 8.2K FLOPs
on ccsd). ILP status 0 (exact) everywhere; build ≤ 2 s at 63 terms.

## Milestone history

- 20261104-201700: dead-code sweep (sum_orbit_clusters superseded).
- 20261104-201217: C-016 orbit-consistent families + signature-class
  splitting. ccsd 1.0717 → 0.9348; ccd 0.9919. C-013 closed.
- 20261104-194244: C-015 canonical-space orbit mining; ccd 0.99994.
- 20261104-192133: B-007 exact lincomb accounting + C-012 sum-orbits;
  adc2 tied gristmill exactly; v5-era relaxed numbers retracted.
- 20261104-190100: B-006 closed. First all-green sym battery.

## Wall-time scoreboard (v3-objective era; measured, not model)

Source: trials/20261105-131316-wallbench.jsonl (+ summary-fix line).
Release binary, (16,64), --layout model, median of 10 after 3 warmup,
interleaved A/B, per-run load recorded (load 2.3 -> 4.8 climbing over the
batch; the late cells — adc2 t16, eom_ccsd — are the most exposed).
Ratio = ours/gristmill median wall time (< 1.0 = we are faster).

| problem | t1 | t16 | model said |
|---|---|---|---|
| mp3   | **0.818** | **0.817** | 0.929 |
| ccd   | **0.636** | 1.063 | 0.954 |
| ccsd  | **0.880** | **0.852** | 0.933 |
| adc2  | **0.621** | 1.019 | 1.001 |
| eom_ccsd (held-out) | 1.137 | 1.303 | (not modeled) |

Model-vs-measured note (the F-004 honesty rule): the model predicted small
wins/parity; measured wins are LARGER at t1 (adc2: 1.001 -> 0.621) and ccd
t16 inverts to a small loss. eom_ccsd loses at both thread counts (load
was highest for those cells; direction looks real, magnitude uncertain).
No model tuning was done against these numbers.
