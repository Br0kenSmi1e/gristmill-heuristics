# CLASSES — expressiveness classes (challenger arm)

One entry per structural class: what it is, measured value, status.
Value = FLOP change on the CCSD doubles definition (63 terms) at (16,64)
unless noted. Trajectory is cumulative; each line assumes all previous
landed classes.

## Landed

- **C-001 product subterm sharing (single level)**
  value: baseline. CCSD 35.47G (level-1 ILP).
- **C-002 multi-level (global) product sharing**
  value: 35.47 → 35.11G. Also exonerated depth-greediness (global ≈
  level-greedy on this corpus) — a finding, not just a cost change.
- **C-003 symmetry quotienting (antisymmetry orbits)**
  value: 35.11 → 25.20G (cost-space). Largest single class.
  Status: sign-correct emission landed for ccd (verified 1.017x);
  ccsd blocked by B-006.
- **C-004 cofactor-family sums (partial keep)**
  value: 25.20 → 23.02G.
- **C-005 cross-cofactor shared sums**
  value: 23.02 → 22.49G. (Motivated by measuring gristmill's candidate:
  7 sum intermediates with 2–4 distinct parents.)
- **C-006 intermediates inside rests**
  value: 22.49 → 21.20G — crossed gristmill (21.42G) on doubles.
- **C-007 term-permutation orbits + cross-output sharing**
  value: ccsd total 21.45 → 16.40G (sym-relaxed); legal no-sym
  22.69G. ADC2: 681.8 → 514M; ccd: 16.74 → 14.60G (sym-relaxed).
  Perm direction bug (B-001) found and fixed here.
  CORRECTION (20261104): the adc2 514M figure was UNDERPRICED — v5-era
  accounting charged nothing for non-identity perm members of
  family-covered reps. The honest adc2 optimum is 687.0M == gristmill.
- **C-012 sum-orbits (pre-sum shared across a perm orbit)**
  value: adc2 689.09 → 686.99M (== gristmill, exact tie); ccd 2.1M;
  ccsd ~19M. Mechanism: reps with identical member (perm, coeff-ratio)
  signatures emit one pre-sum S = sum of rep evals; each orbit element
  rides S via a free permute. Priced exactly in the ILP via cluster
  variables w_C <= z_t. Includes identity-member coeff folding.
  Verified: full battery green (trials/20261104-192133.jsonl).

## Open

- **C-008 sums of sums** (sum nodes whose summands include sum nodes).
  Suspected small; unmeasured.
- **C-009 unsummed-precursor sharing** (form A·B once, use both
  c-summed and c-kept). Unmeasured; check gristmill candidates for
  instances before sizing.
- **C-010 partial-keep families with perm members** — currently only
  identity members join; the permute-based generalization needs the
  keep-set bookkeeping done canonically (not position ids).
- **C-013 honest-optimum residuals (the real gaps, post-B-007)** — with
  exact pricing the ILP optima sit ABOVE gristmill: ccd 16.895G vs
  16.626G (+1.6%), ccsd 23.630G vs 21.422G (+10.3%), adc2 == gristmill
  (tie). By the space-completeness clause these are missing structural
  classes; the old relaxed numbers (0.878/0.766) were underpriced
  relaxations and are retracted. Mechanism unknown; next: structural
  diff of gristmill's ccd/ccsd candidates vs our optima.
- **C-014 adc2 tie-break** — adc2's honest optimum == gristmill exactly;
  the goal needs strictly below, so a class gristmill does not exploit
  is required (C-008/C-009 candidates, or family-level sum-orbits).

## Anti-classes (looked, don't pay)

- MCTS/RL over the move graph: unjustified while the ILP is exact and
  fast at corpus scale. Revisit only if the universe blows up at
  CCSD(T) scale (F-017-blocked anyway).
- E-graph arm: AC+distributivity explosion; demoted (see design doc).
