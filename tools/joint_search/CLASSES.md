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

## Open

- **C-008 sums of sums** (sum nodes whose summands include sum nodes).
  Suspected small; unmeasured.
- **C-009 unsummed-precursor sharing** (form A·B once, use both
  c-summed and c-kept). Unmeasured; check gristmill candidates for
  instances before sizing.
- **C-010 partial-keep families with perm members** — currently only
  identity members join; the permute-based generalization needs the
  keep-set bookkeeping done canonically (not position ids).
- **C-011 legal-space CCSD residual** — no-sym 22.69G vs gristmill
  21.42G; decomposition unknown (could be C-008/C-009 or unseen).

## Anti-classes (looked, don't pay)

- MCTS/RL over the move graph: unjustified while the ILP is exact and
  fast at corpus scale. Revisit only if the universe blows up at
  CCSD(T) scale (F-017-blocked anyway).
- E-graph arm: AC+distributivity explosion; demoted (see design doc).
