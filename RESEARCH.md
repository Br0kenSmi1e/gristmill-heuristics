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
  geomean 1.030x t1 / 1.031x t16 vs same-equation gristmill ON THE DEBUG
  BINARY. Critic follow-up (F-027/F-033, trials/f027-f033-replication.jsonl):
  the only genuine non-fallback result, mp3, replicates on debug
  (1.180-1.186x t1 x3) but is PROFILE-DEPENDENT — on release it shrinks to
  1.017-1.045x t1 and t16 is tree-roll-dependent (F-035 bimodality); the
  debug->release gain is larger for the baseline than for our schedule.
  Net: same-backend PARITY on release, modest win on debug. The v2-priced
  greedy selector beats the baseline only on mp3: search quality under the
  corrected model is the binding Axis-2 question (F-007 reframed); I-007
  selects the M2 algorithm. F-036 owns the release-profile migration.
  ccsd_t: baseline memory-guarded (F-017); the stale selected candidate's
  27.1 s/iter is 100% reduction-class ops by profile (F-029). Exam-cd921bd
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
- **The backend's reduction path is a separate cost regime** — but its
  magnitude is PROFILE-dependent (F-033/F-036): 0.02-0.09 GF/s on debug,
  4.2 GF/s on release (15x below clean GEMM, thread-flat). v1/v2's 600x
  catastrophe was largely a debug artifact. All rates are release-regime
  (calibrate-v2) since model v3.
- **Rates split by cache regime, not just op class** (model v3): slow GEMM
  2.4x faster in-cliff than out (39.7 vs 16.9 GF/s t1); bandwidth splits by
  rank (scattered rank>=5) AND working set (L3-effective 40 MB cliff);
  k-dominant non-clean contractions are permute-bound. Machine: i5-13400F
  6P+4E hybrid — t1 numbers carry a 20-30% P/E-core lottery; several t16
  mid-size classes are unstable (documented CI exceptions).
- **Model v3 program-level validation (release, gristmill-free references)**:
  totals within 1.12-1.58x, per-op spearman 0.89-0.96 (v2 era: 0.37).
  trials/model-v3.jsonl.
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

Model v3 landed on the release regime (33a4e83) and exam-774420a ran on
release: the selector emits the verified baseline on ALL six runnable
problems (Axis-2 parity by construction; mp3's v2-era win does not
survive v3 pricing, consistent with its release measurement). The
milestone is recorded; a fresh-context critic is due (user-launched).
Open queue: F-005/I-006 (enumeration budget — the only remaining route
to spaces bigger than baseline on adc2-class problems), F-035
(execution-tree pinning), F-017 (memory wall), and then M2 codegen.
