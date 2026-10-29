# CONTRIBUTION

## Central claim

**Corrected headline (baseline = gristmill run on the IDENTICAL equations,
exported from the same truthmill references; both sides get our layout
layer):** the HL selector is currently at rough PARITY with gristmill --
geomean 0.90x across six problems (wins: mp3 1.09x/1.18x, ccsd t16 1.09x;
parity: ccd, ccsd_lambda t1; losses: eom_ccsd 0.77x, adc2 0.67x t1). An
earlier version of this document claimed a 1.42x CCSD win; that was an
artifact of a weaker first baseline (equations re-derived through a
different drudge path that led gristmill to a worse local optimum). The
correction was produced by the pipeline's own verification-and-timing
machinery.

What the infrastructure did establish: verified, memory-safe, wall-time
measured scheduling from truthmill IR to tenferro, with a fast structural
proposer (100-1000x faster search, reaches CCSD(T) where kernel enumeration
hangs) -- and a trials record that caught and localized its own failures
(kernel enumeration blowup, flops-greedy memory bombs, a 20x model
mispricing on rank-6 ops, and now the selector gap on eom/adc2).

Supporting evidence for *why* the selector matters: a FLOP-greedy replica of
gristmill's selection policy on the same kernel either stalls after one
firing or accepts a memory-catastrophic intermediate (58 TB estimated peak on
CCD) -- FLOP savings are blind to both structure preservation and memory,
which is precisely what the wall-time model sees.

A wall-time-trained, verification-gated heuristic system for scheduling
symbolic quantum-chemistry tensor equations onto a dense tensor backend
(tenferro-rs) beats the backend's own planner by 1.03–1.45× steady-state wall
time on equation sets the heuristics never saw (held-out: CCSD(T), CCSD-Λ,
EOM-CCSD, ADC(2); geomean ≈1.13× single-thread, ≈1.06× at 16 threads), and by
1.2–3.1× on the training sets (CCD, CCSD). The system was developed as an
instance of Weng's *heuristic learning*: a coding agent edits scheduling
heuristics against tiered feedback, with exact symbolic verification
(truthmill `verify`) plus a numerical random-tensor oracle gating every
candidate — no wrong-but-fast schedule can ever score.

Equally load-bearing are the mechanistic findings about where wall time
actually goes in this stack, which contradict the naive prior:

1. **Physical layout is the dominant single-op lever.** On the dominant CCD
   contraction (v=64, o=16), permuting one operand to GEMM-friendly layout is
   2.7× faster single-threaded than letting the einsum planner handle the
   as-is layout (382 ms → 142 ms; raw 2-D GEMM matches the permuted einsum, so
   the loss is layout handling, not GEMM quality). In the iterative regime,
   layout conversion of loop-invariant integrals hoists out of the timed loop.
2. **Symbolic FLOP counts are not a valid proxy for wall time when the backend
   re-plans.** The un-factorized CCD reference is ~10^5× worse in symbolic
   FLOPs than a greedily factored program at (o=100, v=1000), yet wall time is
   unchanged, because tenferro parenthesizes n-ary einsums internally.
3. **Naive binary factorization *regresses* vs. the backend's n-ary planning**
   (CCSD 0.53×). Factorization is needed only for memory feasibility
   (EOM-CCSD's naive form requires >1 GB backend-internal temporaries for 121
   ops at modest sizes; CCSD(T)'s D3 input alone is 8.6 GB at o=16,v=64).
4. **Multi-use tensors punish per-use-optimal layouts.** Choosing each block's
   layout by its heaviest use regressed CCSD by 1.45×; a global cost model
   (sum over uses, constant mismatch penalty) fixed it.

## Mechanism

Pipeline: truthmill `Problem` IR → program transforms (memory-bounded n-ary
splitting, exact-duplicate CSE) → uniform layout planner (blocks and SSA
intermediates are "storage entities"; each picks among {natural} ∪ {per-use
ideal layouts} the one minimizing Σ_uses weight·penalty) → tenferro
interpreter with prepared einsum plans, liveness-based value freeing, and a
setup/steady-state split matching the iterative regime. The HL loop: trials
logs in `trials/`, regression corpus = numerical oracle on all 7 problems,
failure-driven policy edits documented in RESEARCH.md.

## Difference from prior work

- gristmill/TCE-style compilers optimize symbolic FLOPs with fixed
  heuristics; we optimize measured wall time (and show the two diverge badly).
- GETT/TCCG establish transpose costs matter; we turn layout choice into a
  learned, per-entity scheduling decision on a backend that does not expose it.
- RL-TNCO and FunSearch-style kernel evolution search per-instance orders or
  episodic programs; here a persistent heuristic system transfers across
  equation sets, with exact symbolic equivalence gating rather than tests alone.

## Evidence and controls

- Correctness: all transforms pass `truthmill::verify` (exact expansion +
  canonicalization under declared antisymmetries) AND an independent
  scalar-loop numerical oracle at machine precision on all 7 problems.
- Controls: naive per-op einsum through tenferro's own planner is the
  baseline (i.e., the backend's built-in optimization is what we beat);
  ablations isolate greedy factoring (harmful), per-use argmax layouts
  (harmful), consumer-blind intermediate reordering (harmful), CSE (small
  consistent win).
- Protocol: steady-state per-iteration medians, interleaved A/B reps,
  best-of-N reported; threads pinned via `CpuBackend::with_threads`; memory
  guards (static peak estimate + RLIMIT_AS backstop) after two OOM incidents.
- Transfer: penalty (2.7) fixed on train; held-out never tuned on.
- Raw data: `trials/*.jsonl`.

## Strongest objection

The selector is greedy over the kernel's spaces with our scoring; gristmill's
full pipeline includes phases we don't exercise (its own summation
optimization details), so "we beat gristmill" means "our wall-time-selected
rewrite sequence beat gristmill's FLOP-selected output on these problems at
these sizes on this backend" - not a claim about gristmill on its native
Fortran toolchain. ccsd_t is unimproved (kernel biclique enumeration has no
budget - 24-term definitions explode; needs a cutoff). Single machine, sizes
capped by 14 GB RAM; the model's four rates are calibrated once per thread
count and could drift across shapes. adc2's gains are partly CSE/layout
(factorization found little there - 1 step).

(Previous objections, still standing:) Single machine, small sizes (o<=16,
v<=64); layout optimization with analytic models is not new (TCE lineage);
residual novelty is backend-behavior modeling + verify-gated search +
transfer evaluation + the HL maintenance loop.

The wins are modest and single-machine, at small sizes (o≤16, v≤64) dictated
by 14 GB RAM; at production sizes GEMM may dominate enough to shrink layout
effects (or permutes may hurt more — untested). The baseline, though real, is
not the strongest conceivable: a gristmill-FLOP-optimized schedule emitted to
the same backend might capture part of the gain. And "heuristic learning"
here is a dozen agent-driven edits with small parameter sweeps — not yet a
continually self-improving system; the eom_ccsd t16 regression (0.89×) shows
the constant-penalty cost model is wrong where small ops dominate at high
thread counts.

## Remaining conceptual obstacle

Layout, factorization, and parallelism decisions interact: splitting an n-ary
op creates intermediates whose layouts are new decisions; the true per-op cost
function depends on operand layouts, sizes, and thread count in a way a binary
match/mismatch model cannot express. The plausible next mechanism is a
measured per-op cost model (profile each expensive op under layout variants,
fit the penalty surface) feeding the global assignment — plus blockwise
symmetry exploitation, the largest untouched FLOP/memory lever.

## Reproduction

```
cargo build --release
# oracle: all 7 problems, machine precision
for p in benches/chemistry/*.json; do
  ./target/release/harness check $p --sizes 0=3,1=4 --transform cse-only --layout v2
done
# headline comparison (train example):
./target/release/harness time benches/chemistry/ccsd.json --sizes 0=16,1=64 \
  --threads 1 --transform cse-only --layout none   # baseline
... --layout v2                                    # policy
```

trials/ holds raw records; RESEARCH.md holds the development narrative and
negative results. Memory guards: `--max-mem-mb` (default 6000), `H_RLIMIT_GB`
(default 10). Do not raise them on this box.
