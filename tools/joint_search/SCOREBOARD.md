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
