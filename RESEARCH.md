# RESEARCH — current understanding

Synthesis, not a log. Raw history: `trials/*.jsonl` (current),
`archive/trials-v1/` + tag `hl-study-final` (prequel study).

## Objective

A compiler: truthmill Problem IR (tensor equations + declared symmetries)
-> verified, memory-safe, standalone tenferro code, optimized for
steady-state wall time. Full acceptance criteria: SPEC.md. How we work:
PROTOCOL.md. What is broken: FAILURES.md. What we might try: IDEAS.md.

## What the prequel established (durable knowledge)

- **Layout, not FLOPs, dominates the FLOP-vs-wall-time gap** on this
  backend (H2 confirmed by profiling; permute/reshape traffic rivals GEMM).
- **A white-box analytic cost model of the backend works**: walk the
  backend's own contraction tree, price clean vs slow GEMM + permute passes
  with calibrated rates; per-op ranking rho ~ 0.93 on ccd/ccsd. This model
  is the compiler's heart — and F-004 is its known hole.
- **Wall-time-priced selection is a real lever**, and the exam-cd921bd
  milestone (debug binary pinned, per-run load, Axis-0 14/14 pairs green):
  **Axis 1 is green on 6/7 problems** (mp3 0.969x genuinely below baseline;
  ccd/ccsd/lambda/eom/adc2 at 1.0 by fallback identity; ccsd_t's stale
  v1-era candidate is RED at 40.6x cap, execution-free check,
  trials/f029-ccsd_t-axis1.jsonl). Axis 2 six-problem
  geomean 1.030x t1 / 1.031x t16 vs same-equation gristmill — but five of six
  cells are fallback-vs-baseline noise (+-9%); the only genuine non-fallback
  result is MP3 (1.194x t1 / 1.139x t16, fewer FLOPs AND faster). The v2-priced
  greedy selector beats the baseline only on mp3: search quality under the
  corrected model is now the binding Axis-2 question (F-007 reframed).
  ccsd_t remains untimed-vs-baseline (F-017; its selected candidate runs but
  at 27.1 s/iter t1). Exam-cd921bd
  numbers supersede a8bf146's (which never pinned its binary profile; its
  absolute ms are unanchored, ratios still valid).
  All 12 mixed per-cell results, including CCSD/lambda thread inversions,
  are reported in F-007. That exam's Axis-1 red led to a mandatory emission
  cap with verified-baseline fallback (F-018); search-quality losses remain
  open. CCSD(T) timing is blocked by F-017's 6 GB harness cap.
- **Composition of levers**: factorization (gristmill kernel) and layout
  are complementary; both are needed.
- **Memory is a feasibility constraint**, not just performance (naive EOM
  needs >1 GB of temporaries; CCSD(T) D3 input alone is 8.6 GB at (16,64)).
- **The backend's reduction path is a third cost regime** (M1.1 grid,
  profiles/calibrate-v1.json): contractions with output rank 0/1 run at
  0.02–0.09 GF/s t1 — even a rank-1 dot product — with no t16 scaling
  (negative for rank-1 output). Matricizable contractions run at the
  calibrated clean/slow GEMM rates; low-rank n-ary chains are fine.
  Splitting n-ary reductions into binaries does not rescue them.
- **Measurement hygiene is load-bearing**: one contaminated batch cost a
  full re-measurement cycle; one weak baseline fabricated a 1.42x "win".
  Both are now protocol rules, and the load guard is structural inside the
  timing commands (F-006).
- **Oracle inputs must satisfy the full symmetry-group orbit**, not only
  direct relations among used blocks. Composed relations through unused
  intermediate access patterns are required by canonicalization (F-009).

## Open questions (ordered)

1. F-005 / I-006: budget kernel enumeration for definitions above the
   20-term envelope. F-008 fixed CCD; CCSD's 63-term structural fallback
   failed both wall-model acceptance and FLOP-mode exact equivalence, so the
   budget must be inside the kernel.
2. F-007 / F-019: ADC2 greedy-path failure is diagnosed, I-005 is retired on
   Axis 1; EOM and thread-inversion mechanisms remain open.
3. F-017: CCSD(T) execution is a HARD RESOURCE BLOCKER. Backend allocations
   exceeded physical memory and filled swap; no full-size reruns. Develop
   memory-safe schedules only on tiny fixtures until static peak is <6 GB.
4. F-016: price the remaining model-gap classes (k-heavy low-rank-output
   contractions at 0.26 GF/s, single-k rank-4-out at 0.9 GF/s, rank-6
   lincomb fixed overhead) via calibration-grid extension; evaluate model v2
   on the held-out ADC2 gap only at a milestone.
5. IDEAS.md I-001..I-004, in that order.

## Next investment

Milestone after exam-cd921bd: EXAM done (green Axis 1, geomean 1.030/1.031),
TOURNAMENT disposition recorded (no evaluable challenger; slot unchanged),
CRITIC pending (fresh-context session). Then: M1 tail = calibration-grid
extension for the F-016 gap classes (k-heavy low-rank-out contractions at
0.26 GF/s, rank-6 lincomb floor); M2 = F-005/I-006 enumeration budget and
v2-era search quality (F-007 reframed).
