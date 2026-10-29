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
- **Wall-time-priced selection is a real lever** (FLOP-greedy selection on
  the same rewrite space loses or stalls), but the valid a8bf146 exam gives
  six-problem geomean 0.902x t1 / 0.883x t16 vs same-equation gristmill.
  All 12 mixed per-cell results, including CCSD/lambda thread inversions,
  are reported in F-007; Axis 1 is red on all six (F-018). CCSD(T) timing
  is blocked by F-017's 6 GB harness cap.
- **Composition of levers**: factorization (gristmill kernel) and layout
  are complementary; both are needed.
- **Memory is a feasibility constraint**, not just performance (naive EOM
  needs >1 GB of temporaries; CCSD(T) D3 input alone is 8.6 GB at (16,64)).
- **Measurement hygiene is load-bearing**: one contaminated batch cost a
  full re-measurement cycle; one weak baseline fabricated a 1.42x "win".
  Both are now protocol rules.
- **Oracle inputs must satisfy the full symmetry-group orbit**, not only
  direct relations among used blocks. Composed relations through unused
  intermediate access patterns are required by canonicalization (F-009).

## Open questions (ordered)

1. F-018: restore Axis 1; only ADC2 has a verified FLOP-gap mechanism, so
   MP3/CCD/CCSD/lambda/EOM need per-schedule diagnosis before more selection.
2. F-008 / F-005: replace lossy definition chunking with budgeted kernel
   enumeration that preserves cross-chunk factorization.
3. F-007 / F-019: ADC2 greedy-path failure is diagnosed, I-005 is retired on
   Axis 1; EOM and thread-inversion mechanisms remain open.
4. F-017: make the CCSD(T) memory cap explicit and consistent with the 14 GB
   machine envelope without weakening allocation safety.
5. F-004 / F-016: price high-rank/strided GEMM and close the ADC2 gap error
   (measured range 1.60–1.92x vs 1.14x predicted; M1).
6. IDEAS.md I-001..I-004, in that order.

## Next investment

M1: calibration suite (rank/shape/stride/thread grid -> versioned rates
file) + cost model v2 + model-vs-measured CI + load guard inside the
timing command. Acceptance: ccsd_t intermediate ops predicted within 2x —
the exact failure F-004 records.
