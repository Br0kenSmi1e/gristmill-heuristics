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
the HL loop invests in any heuristic surface. **Result (see Current state):
H2 wins decisively.**

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
## Milestone 2 results (layout v2 = uniform cost model over storage entities)

Blocks and intermediates are uniform "storage entities"; each picks the
layout minimizing sum over uses of weight * (1 if ideal else penalty).
Machinery: liveness-based freeing, static memory guards + RLIMIT backstop,
factmem transform (split only >256MB-union n-ary einsums).

Measured ((16,64) unless noted, best-of-N medians):
- Train: ccd 1.39x t1 / 3.1x t16; ccsd 1.38x t1 / 2.8x t16; mp3 ~1.0x.
- Held-out (penalty=2.7 fixed): adc2 1.44x t1 / 1.10x t16; ccsd_lambda
  1.00x / 1.18x; ccsd_t (8,32) 1.06x / 1.11x; eom_ccsd 1.03x / **0.84x t16
  REGRESSION**.
- Negative results that shaped the design: (a) full greedy n-ary splitting
  REGRESSES CCSD to 0.53x (backend plans small n-ary ops better; split only
  for memory); (b) per-use argmax layout (v0) regresses CCSD 1.45x;
  (c) consumer-blind intermediate reordering doubles LinComb traffic.
- Memory is a feasibility constraint, not just performance (H3 confirmed):
  naive EOM-CCSD needs >1GB backend-internal temporaries for 121 ops at
  (16,64); CCSD(T) D3 input alone is 8.6GB at (16,64) -> ran at (8,32).
- Thread-count changes the cost landscape: v2 wins shrink or invert at t16
  (memory-bound); penalty must be thread-conditioned.
## HL loop iterations completed (see trials/ for raw data)

1. v0 per-use argmax layout -> CCSD 1.45x regression; diagnosed: multi-use
   starvation.
2. v1/v2 uniform entity cost model (blocks + intermediates, consumer-aware):
   fixed v0 failure; exposed that full greedy splitting itself regresses CCSD
   (0.53x) because the backend plans n-ary einsums well -> split only for
   memory (factmem).
3. Liveness freeing: fixed OOM, also large speed win (naive ccsd t16
   1689 -> 743 ms).
4. Penalty sweep on train: flat 2.7 fine; thread-conditioned penalties NOT
   supported by data (within noise). Weak-signal learning; noted.
5. cse_exact (exact-duplicate sharing; references contain up to 4x duplicate
   contractions): small consistent win (ccsd t16 659 -> 627 ms).
6. Final policy: {transform: cse-only / factmem-when-needed, layout v2,
   penalty 2.7}. Held-out: adc2 1.42x/1.15x, ccsd_lambda 1.03x/1.11x,
   ccsd_t 1.06x/1.10x, eom_ccsd 1.04x/0.89x (t1/t16). Geomean 1.13x t1,
   1.06x t16. Residual failure: eom_ccsd t16 (small-op layout sensitivity at
   high thread count; constant-penalty model inadequate there).

## External comparison (gristmill, same backend, verified candidates)

drudge-derived, gristmill-optimized CCD and CCSD schedules imported as
truthmill candidate Programs (tools/export_gristmill_*.py); both pass
truthmill::verify exactly and the numerical oracle. Interleaved timing at
(16,64), ms, t1/t16:

- CCD: naive 925/195; ours (CSE+layout) 601/169; gristmill 681/154;
  gristmill+layout 395/121.
- CCSD: naive 1903/619; ours 1528/589; gristmill 995/327; gristmill+layout
  659/266.

Reading: factorization and layout are complementary levers; composition wins
everywhere. Our layout policy adds 1.2-1.7x on top of gristmill's FLOP-optimal
schedules; on CCD ours alone beats gristmill alone at t1 despite 1.7x more
FLOPs. CCSD exposes our missing piece: cross-term shared intermediates
(gristmill executes ~5x fewer FLOPs than our per-op greedy there).

Also fixed en route: oracle input generation now enforces cross-block symmetry
relations (one projected representative per symmetry component) -- the
gristmill candidate exposed that realism bug by failing the oracle despite
passing verify.

## HL search loop (M3): measured-acceptance factorization search

New machinery: `share_pairs` (canonical pair signatures -> shared
intermediates) and `harness search`: forward selection where each candidate
intermediate is introduced only if MEASURED wall time improves (0.5%
threshold), each step gated by truthmill::verify; artifacts saved as
candidate Program JSONs.

Preliminary (CCD/CCSD, (16,64)):
- Blind FLOP-greedy sharing HURTS CCSD (2233 vs 1529 ms t1) -- same lesson as
  greedy splitting: the model must not outrun the measurement.
- Measured-acceptance search improves our policy: CCD 601 -> 508 ms t1,
  CCSD 1529 -> 1269 ms t1 (both with layout v2 on top). Still behind
  gristmill+layout (392 / 657): pair-sharing lacks (a) sum factorization
  (A*B+A*C -> A*(B+C)), (b) multi-step lookahead, (c) chained intermediates
  (pairs involving freshly-created intermediates).
- LAYOUT-AWARE acceptance (evaluating candidates with layout planning
  applied) showed an inconsistent baseline (reference+layout timed 3886 ms
  in-loop vs 1384 via the time path) -- UNRESOLVED, possibly load-related;
  investigate before trusting that variant.

CONTAMINATION WARNING: from ~10:44 the box ran ~20 user julia jobs
(load avg 24, swap touched). All timings after that point are unreliable
in absolute value (interleaving keeps A/B fair-ish). Re-run key comparisons
when the machine is free: share2/share3 need confirmation.

## Next investment

Measured per-op cost model (profile expensive ops under layout variants; fit
penalty surface as function of op size x threads) replacing the binary
penalty; then blockwise symmetry exploitation (the largest untouched lever:
~4x FLOPs+memory on antisymmetric tensors); cross-term shared intermediates
beyond exact duplicates (structural CSE up to permutation); size-ladder
validation at larger v/o on a bigger machine; standalone Rust codegen for
winning schedules (interpreter and emitter share the schedule IR).
