# Heuristic learning for quantum-chemistry tensor code generation

## Objective

Build and study a **heuristic system (HS)** — in the sense of Weng's *Learning
Beyond Gradients* — that turns symbolic tensor expressions (truthmill `Problem`
IR) into tenferro-rs execution schedules, optimized for **steady-state
per-iteration wall time** in iterative many-body workloads (CC-style). The
update mechanism is a coding agent editing scheduling heuristics; feedback is
tiered (symbolic cost → small-size wall time → target-size wall time);
correctness is gated by exact symbolic verification (truthmill `verify`) plus a
numerical random-tensor oracle. Generic at least for quantum chemistry:
new equation sets drop in as data, heuristics must transfer.

## Why this may matter (hypotheses about where wall-time headroom lives)

- **H1 — factorization/FLOPs dominate.** Then the problem is gristmill-class
  and wall-time awareness adds little.
- **H2 — layout/permutation overhead dominates the FLOP-vs-wall-time gap.**
  Tenferro is dense column-major; index ordering of every stored tensor decides
  GEMM-friendliness. GETT/TCCG literature shows transposes can rival GEMM cost.
- **H3 — memory footprint / intermediate materialization dominates** at QC
  sizes (this box: 14 GB RAM; dense v⁴ at v≈180 ≈ 10 GB).
- **H4 — per-op overhead dominates** (unlikely at QC sizes; a measurement trap
  at toy sizes).

Milestone 1 includes a profiling experiment on CCSD/CCD to rank H1–H4 before
the HL loop invests in any heuristic surface.

## Architecture

```
truthmill Problem JSON (corpus: mp3, ccd, ccsd, ccsd_t, eom_ccsd, adc2, ccsd_lambda)
  → scheduler (HEURISTIC POLICY — the module HL edits):
      factorization, contraction order, per-tensor layout,
      per-tensor symmetry storage scheme, thread budget
  → schedule IR → interpreter (tenferro eager/prepared plans; fast HL loop)
               → codegen (standalone Rust source; deliverable artifact)
  → feedback: truthmill symbolic_flops/log_flops → predicted permute/bytes
              → small-size wall time → target-size wall time (protected metric)
  → trials/*.jsonl + regression corpus + periodic compression (HL memory)
```

Key decisions (agreed with user):

- **Iterative regime is primary**: prepare plans once, time steady-state
  iterations; one-shot metrics secondary.
- **Symmetry is structural, staged**: schedule IR carries per-tensor storage
  scheme {dense-full, antisymmetrizer-pattern, blockwise-compressed} from day
  one; v1 executes schemes 1–2; blockwise gated on H3 evidence. "When does
  symmetry exploitation pay" is itself a heuristic question (FLOP savings vs
  GEMM fragmentation) — a target for learned heuristics, not a fixed rule.
- **All-Rust pipeline**; truthmill as git dependency; tenferro 0.7.1 pinned and
  treated as fixed (upstream PRs possible later but out of scope now).
- **HL loop does not invoke rustc per candidate**: harness interprets schedules
  via tenferro prepared einsum plans; codegen emits source only for winners.

## Correctness gates (non-negotiable)

1. `truthmill::verify`: exact symbolic equivalence of any candidate `Program`
   against the trusted reference (expansion + canonicalization under declared
   symmetries).
2. Numerical oracle: independent scalar-loop evaluator vs. tenferro
   interpreter on random symmetry-respecting inputs at tiny sizes.
3. Known gap (flagged by truthmill corpus README): no cross-check of the
   reference equations against a real QC implementation (e.g., PySCF). Our
   gates check schedule-vs-reference, not reference-vs-physics. Input blocks
   are generated independently per domain pattern (within-block symmetry
   projected), so cross-block symmetry relations are not exercised — a threat
   to physical realism, not to schedule correctness.

## Evaluation protocol (protected)

- **Baselines**: (i) naive per-op einsum through tenferro's own planner;
  (ii) FLOP-greedy factorization mechanically scheduled.
- **Train/held-out split**: develop heuristics on {mp3, ccd, ccsd};
  hold out {ccsd_t, eom_ccsd, adc2, ccsd_lambda}. Heuristics must never see
  held-out sets during development.
- **Sizes**: ladder (o,v) ∈ {(8,40),(16,80),(32,96)} subject to 14 GB RAM;
  report scaling, not single points. Threads ∈ {1, 16}, pinned.
- **Metric**: steady-state per-iteration wall time, median of ≥10 iters after
  ≥3 warmup; geometric-mean speedup vs. baselines; no regression on any
  training member beyond tolerance.
- Machine-specific results; second-machine check before general claims.

## Prior art (positioning)

- gristmill/drudge: FLOP-driven factorization + codegen; fixed heuristics.
- TCE (NWChem), TCCG/GETT (Springer/Bientinesi): transpose cost is real; fixed
  compilers.
- RL-TNCO (Meirom et al. 2022): RL for contraction *order*, FLOP objective,
  per-instance, no factorization/layout/wall-time.
- FunSearch / kernel-evolution (Kernel-Smith, KernelFoundry, EvoEngineer):
  LLM+evaluator program evolution; episodic, no persistent verified memory.
- Weng HL: continual maintenance of a heuristic system with explicit memory,
  regression tests, compression. Our differentiator vs all: wall-time-trained,
  symmetry-aware scheduling heuristics with exact symbolic verification and a
  protected transfer evaluation.

## Current state

- Repo scaffolded; corpus vendored from truthmill examples/chemistry.
- Harness under construction: load Problem → instantiate inputs (per-block,
  within-block symmetry projection) → interpret reference via tenferro →
  oracle check → steady-state timing.

## Next investment

Milestone 1: harness `check` + `time` on mp3/ccd/ccsd; naive baseline numbers;
CCSD profiling experiment (factorization/layout/order variants) to rank H1–H4.
That ranking decides which heuristic surface the HL loop attacks first.
