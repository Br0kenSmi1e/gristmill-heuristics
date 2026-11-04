# SCOREBOARD — verified numbers (challenger arm)

Source: trials/20261104-192133.jsonl (full battery, all gates green).
All candidates emitted by the global arm, verified by truthmill::verify +
numerical oracle at (3,5); FLOP ratios at (16,64) vs the same-equation
gristmill baselines (benches/candidates/*_gristmill.json).

## Verified scoreboard (sym ON; sym is a no-op on mp3/adc2)

| problem | ratio | goal (<1.0) | honest ILP optimum | gap lever |
|---|---|---|---|---|
| mp3   | 0.6566 | PASS | 11.283G (opt == emitted) | — |
| ccd   | 1.0161 | fail | 16.895G vs grist 16.626G | C-013 (+1.6% space gap) |
| ccsd  | 1.1031 | fail | 23.630G vs grist 21.422G | C-013 (+10.3% space gap) |
| adc2  | 1.000000 | fail (tie) | 686,989,824 == grist | C-014 (need a class gristmill lacks) |

"Honest ILP optimum" = cost-space optimum with exact (post-B-007)
pricing; emitted == optimum on all four (realization is exact now).
The pre-B-007 relaxed numbers (ccd 0.878x, ccsd 0.766x, adc2 0.748x)
were underpriced relaxations and are RETRACTED.

## Milestone history

- 20261104-192133: B-007 (exact lincomb accounting) + C-012 (sum-orbits)
  landed; adc2 tied gristmill exactly; v5-era relaxed numbers retracted
  as underpriced.
- 20261104-190100: B-006 closed (formation orbit signs + perm-member
  double-emit). First all-green battery with sym on all four problems.
- 20261104-181011: pre-B-006 state (ccsd verify failing).
