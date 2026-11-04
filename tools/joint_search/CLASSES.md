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
  ccsd ~19M. Mechanism: items with identical orbit signatures (groups,
  then also family results after C-016) emit one pre-sum; each orbit
  element rides it via a free permute. Priced exactly in the ILP via
  cluster variables w_C <= item-chosen. Includes identity-member coeff
  folding. Verified: full battery green.
- **C-015 canonical-space orbit mining** — canon_full now takes the
  exact joint orbit min (label bijections × per-factor signed orbits);
  the old original-space mining was not canonical (mining does not
  commute with relabeling) and hid sharing (ccd formed duplicate
  intermediates). Made fast via order-statistic monotonicity (separable
  per-factor orbit-min per renaming) plus removal of the dead
  factor-permutation loop. ccd 1.0161 → 0.99994; mp3_sym solve
  17.1s → 0.59s.
- **C-016 orbit-consistent families** — a family whose covered reps all
  share one orbit signature emits one result; the whole perm orbit rides
  it via free permutes (gristmill's ccsd V67 pattern: 20 terms from one
  T3-combine + 3 permutes). Family candidates are split into signature
  classes (one inconsistent rep no longer poisons a family). ccsd
  1.0887 → 0.9348; ccd → 0.9919. Subsumes/retires C-010 (perm members
  never enter the S-sum at all) and closes C-013.

## Open

- **C-008 sums of sums** (sum nodes whose summands include sum nodes).
  Suspected small; unmeasured.
- **C-009 unsummed-precursor sharing** (form A·B once, use both
  c-summed and c-kept). Measured on adc2's chosen solution: ZERO
  instances (no chosen block recurs with a different keep set).
- **C-014 adc2 tie-break [BLOCKER]** — adc2's ILP-exact optimum is
  686,989,824 == gristmill to the FLOP (identical cost histograms; the
  space contains gristmill's program and nothing cheaper). adc2 has no
  declared symmetries, so no sym lever. Measured anti/zero-value classes
  on adc2: multi-operand fused steps (uniformly worse: 13.4G vs 648M),
  value-level coincidences (none), C-009 (none). The goal's strict-beat
  clause needs a class with positive value on adc2; none named yet.
  Next hypotheses: sums-of-sums inside rests (C-008 proper), or accept
  the tie and report to the mainline tournament.
- **C-017 multi-operand steps [ANTI-CLASS]** — measured: fused n-ary
  pricing (k-1+contr)·p is uniformly worse than binary DP on this
  corpus. Do not pursue.

## Anti-classes (looked, don't pay)

- MCTS/RL over the move graph: unjustified while the ILP is exact and
  fast at corpus scale. Revisit only if the universe blows up at
  CCSD(T) scale (F-017-blocked anyway).
- E-graph arm: AC+distributivity explosion; demoted (see design doc).
