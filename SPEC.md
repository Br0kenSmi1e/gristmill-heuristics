# SPEC — what "the compiler" must do

## Layer 0 — positioning (changed only by dated amendment)

Given the same tensor equations (truthmill Problem IR), produce **verified**
standalone tenferro code with the best steady-state wall time we can
measure. We do NOT compete on equation derivation (SeQuant/pdaggerq own
that). gristmill appears in this spec ONLY as the tier-1 baseline we must
beat — the strongest existing FLOP optimizer, imported on identical
equations. What factorization engine sits inside our compiler is NOT a
constitutional question: it is a contested slot (current incumbent: the
gristmill-rewrites kernel, because it works and is verified; challengers:
IDEAS.md entries), decided by tournaments on Axis 1/2 numbers like every
other slot. Our contested ground: backend-aware scheduling (layout,
permute, thread regime), verified correctness, memory safety, and symmetry
storage. New algorithms are welcome — they enter through IDEAS.md with
pre-registered adoption rules, and win or lose on the axes below.

## Corpus and protected split

- Train: {mp3, ccd, ccsd}. Held-out: {ccsd_t, eom_ccsd, adc2, ccsd_lambda}.
  Drop-in (M4): CC2 or ADC(3), zero heuristic changes allowed.
- Sizes: (o,v) in {(8,40), (16,64), (32,96)} subject to 14 GB RAM; ccsd_t
  at (8,32) due to input footprint. Threads in {1, 16}, pinned.
- Metric: steady-state per-iteration wall time, median of >=10 after >=3
  warmup, clean machine under the load guard.

## Axes (each must be executable; commands are current harness names)

- **Axis 0 — correctness (gate).** Every emitted schedule passes
  `harness check` (truthmill::verify + numerical oracle,
  symmetry-respecting inputs). 100% pass, no flag to disable.
- **Axis 1 — FLOP quality (gate).** `harness flops` of our schedule
  <= 1.05x the same-equation gristmill baseline's count, every corpus
  problem. This is a REAL gate, not a freebie: it holds for the incumbent
  engine by construction but constrains any challenger that displaces it.
- **Axis 2 — backend-controlled schedule quality.** Same-equation gristmill
  baseline (`tools/gristmill_baseline.py`) imported as a candidate and run
  through OUR layout+memory stack via `harness time --candidate ...` vs our
  full pipeline. Target: geomean >= 1.15x, no single loss beyond 2%.
  CURRENT STATE: FAIL — geomean 0.90x t1 (F-007).
- **Axis 3 — system-vs-system.** gristmill's emitted Fortran+BLAS vs our
  emitted tenferro binary, same machine. Report achieved GF/s alongside ms
  to decompose compiler-vs-backend. Target: parity at t16. (Blocked on
  codegen, M3.)
- **Axis 4 — generality.** Drop-in problem: import a foreign-syntax
  equation set, produce verified competitive code, zero heuristic changes,
  within one day. (M4.)
- **Axis 5 — symmetry storage.** On problems with declared antisymmetry:
  >=2x peak-memory reduction with <10% wall-time penalty, oracle passing on
  symmetry-respecting inputs. (M5.)

## Baselines (tiered)

1. gristmill, same-equation import — exists (`tools/gristmill_baseline.py`).
2. pdaggerq, SeQuant — equation-level import as candidates. (M4.)
3. TCE/OpMin — literature FLOP/memory numbers only; their regime
   (out-of-core) does not overlap our dense in-core envelope. Say so; do
   not fake-compare.

## Milestones

- M1: calibration suite + cost model v2 (rank/stride-aware slow path) +
  model-vs-measured CI gate. Closes F-004.
- M2: budgeted hybrid search + measured final acceptance; Axis 2 green on
  the full corpus incl. ccsd_t. Closes F-005, F-007.
- M3: codegen to standalone Rust for all 7 problems.
- M4: second machine + pdaggerq/SeQuant imports + drop-in test. Axes 3, 4.
- M5: blockwise symmetry storage. Axis 5.
