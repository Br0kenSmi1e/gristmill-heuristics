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

Analytic cost models for tensor contraction scheduling (the approach adopted
for the next optimizer iteration):
- TCE lineage: Gao et al. PPoPP'05 (out-of-core perf model); JPDC 2012 /
  LCPC'04 "empirical performance model-driven data layout optimization and
  library call selection for tensor contraction expressions" (closest prior
  art: analytic layout-selection model, empirically calibrated); Pitzer et
  al. JPCA'09 (algebraic/CSE flop minimization); Space-Time Trade-Off TR.
- TCCG (Springer/Bientinesi TOMS'18): per-contraction model choosing
  TTGT/GETT/LoG; transpose cost as data movement.
- COGENT (Kim et al. CGO'19): analytic DRAM-transaction model ranking kernel
  configs; beats hours-long autotuning (Tensor Comprehensions) on CCSD(T)
  contractions - direct evidence that analytic ranking suffices in-domain.
- Roofline (Williams'09); CTF/Solomonik communication models (parallel).

What they do not cover (our residual novelty): (1) they model hardware
directly; we must model a self-planning BACKEND (tenferro re-plans
internally) - the model's terms describe the backend's lowering behavior;
(2) none gate candidate schedules with exact symbolic equivalence;
(3) whole-equation-set SSA scheduling with iterative-regime layout hoisting
and protected transfer evaluation across methods. JPDC 2012 means "first
analytic layout model" is NOT claimable.

Earlier positioning notes:

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
## White-box analytic cost model (v3 policy) and clean validation

Model: per op, walk the backend's OWN contraction tree (public
ContractionTree/step_plan API); per step, clean vs slow GEMM rate by whether
physical operand layouts equal the plan's canonical targets, plus permute /
reduce / final-permute passes at bandwidth, floored by dispatch. Four
calibrated rates per thread count (microbench). Per-op ranking vs measured:
Spearman rho ~0.93 on ccd/ccsd. Layout assignment by iterated conditional
modes over {natural}+per-use ideals with local cost deltas.

Clean-machine results (Julia fleet paused; earlier batches had contention
contamination up to 15x - methodology note: check load before benchmarking):
- Model-driven layout FIXED the t16 regression pathology of the
  constant-penalty heuristic (ccsd t16: v2 0.91x -> model 1.02x; eom t16:
  0.89x -> 1.02x). Mechanism confirmed: thread-scaled rates in the model.
- ccd/ccsd, cse+model vs naive: 1.30x/1.02x, 1.23x/1.02x (t1/t16).
- gristmill composition: ccd g+model 449/139 (1.96x/1.60x vs naive);
  ccsd g+model 1043/363 (1.85x/2.20x). Layout adds 1.04-1.5x over gristmill.
- Held-out (model, untuned): adc2 1.26/1.62, ccsd_lambda 0.96/1.12,
  ccsd_t 1.07/1.14, eom 1.03/1.02; geomean 1.07 t1 / 1.21 t16.

## Next investment

Cross-term factorization search (beam over verified rewrites, priced by the
analytic model - the model is what makes factorization search possible at
all, since candidates evaluate in microseconds and extrapolate to unrunnable
sizes). Target: close the CCSD gap (gristmill executes ~5x fewer FLOPs via
shared intermediates) without importing gristmill. Then blockwise symmetry.

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


NOTE (reconciliation, clean machine): the M3/share* timings above were
contaminated by concurrent load and are SUPERSEDED by trials/clean_*.jsonl.
The share_pairs + measured-acceptance search machinery survives; its numbers
need re-measurement. The analytic model now allows MODEL-priced acceptance,
which makes the search inner loop microseconds instead of wall-clock runs.

## HL selector on gristmill-rewrites kernel (the converged path)

Architecture: truthmill Problem -> bridge (SSA expansion to sum-of-products)
-> gristmill-rewrites State -> greedy selector over kernel rewrite spaces
(biclique / parenthesize / permutation), tiered scoring (kernel log_flops
prefilter top-8 -> our analytic wall-time model WITH layout-in-the-loop) ->
verify-gated -> candidate Program JSON -> tenferro execution. Search cost:
~30 s for CCSD (16 steps, all biclique firings incl. chained intermediates).

Results (clean machine, (16,64) unless noted, best-of-N, t1/t16 ms):
- vs naive backend planner: ccd 2.03x/1.87x; ccsd 2.62x/3.03x; ccsd_lambda
  2.33x/2.64x; eom_ccsd 5.78x/4.38x; adc2 2.08x/1.59x.
- vs gristmill-FLOP-optimized + our layout (previous best): ccd 449/139 ->
  434/119; ccsd 1043/363 -> 736/263 (1.42x/1.38x BETTER). The wall-time-priced
  selector beats FLOP-greedy selection on the same rewrite space.
- All selected candidates pass truthmill::verify AND the numerical oracle.
- Limitation: ccsd_t stalls - the kernel enumerates ALL maximal bicliques
  with no budget (24-term def -> explosion). Needs a kernel enumeration
  cutoff (gristmill's greedy_cutoff analogue); selector currently skips
  defs with >20 terms.

Design validation: the "selector replaces FLOP-greedy selection, kernel owns
generation" hypothesis is CONFIRMED. Tiered scoring (symbolic prefilter ->
wall-time model) mirrors the HL feedback tiers inside the selector itself.

## Structural proposer (fast path) + the ccsd_t model failure it caught

Second selector engine: signature-based pair-sharing proposals (O(T*F^2)
hashing, no enumeration), model-priced (~1 ms/proposal), kernel-verified per
fired step (equivalent_states), truthmill::verify + oracle at the end. Search
wall time: ccd 0.04 s, ccsd 1.1 s, ccsd_t 5.4 s, eom 6 s (vs 30 s+ and hangs
for the enumeration path). REACHED ccsd_t (enumeration cannot).

Measured quality (candidate + model layout, t1/t16 ms):
- ccd 563/171 (kernel-selector: 434/119); eom 2839/1271 (vs 1175/491);
  ccsd_lambda 1212/626 (vs 650/358); adc2 70/81 (vs 101/87 - BETTER);
  ccsd 1812/669 (vs 736/263); ccsd_t 8075/4250 - SLOWER than naive
  (5051/3706): the model predicted 0.41 s, a 20x error.
- Diagnosis (per-op profile of the ccsd_t candidate): the fired
  intermediate-forming rank-6 ops run at ~0.6 GF/s - 100x below clean GEMM,
  37x below the modeled slow-path rate. The binary clean/slow layout
  classification is too coarse for high-rank operands. Model needs a
  rank/shape-dependent slow-path term. THIS is the HL loop working as
  designed: trace shows verify_reject events + the pricing failure is
  localized, quantified, and actionable.
- Also learned: model-only acceptance can ship regressions (ccsd_t). Either
  measured spot-checks or a better slow-path model is required before
  trusting fast selection on new shapes.

## Superseded earlier notes (kept for audit)