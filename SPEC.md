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
- Regime note (user decision): sizes are bounded by this machine (14 GB,
  16 cores). All claims are machine-relative; the spec measures the
  compiler, not the machine. Reaching beyond the in-core envelope
  (tiling/spill/GPU) is out of scope UNTIL Axis 5 evidence says memory is
  the binding constraint — at which point the envelope itself becomes the
  target.

## Axes (each must be executable; commands are current harness names)

- **Axis 0 — correctness (gate).** Every emitted schedule passes
  `harness check` (truthmill::verify + numerical oracle,
  symmetry-respecting inputs). 100% pass, no flag to disable.
- **Axis 1 — FLOP quality (gate).** `harness flops` of our schedule
  <= 1.05x the same-equation gristmill baseline's count, every corpus
  problem. This is a REAL gate, not a freebie: it holds for the incumbent
  engine by construction but constrains any challenger that displaces it.
- **Axis 2 — same-backend decomposition (DIAGNOSTIC, not acceptance).**
  Same-equation baseline candidates run through OUR stack on tenferro
  (`harness time --candidate ...`). Not a spec axis (user decision:
  backends are part of each system; the product comparison is Axis 3).
  Kept as the attribution instrument: when Axis 3 shows a loss, this
  decomposes it into compiler-vs-backend. Also the natural development
  gate pre-M3, while our codegen does not exist yet.
- **Axis 3 — system-vs-system (THE competitive axis).** Each baseline runs
  its own backend in its best native configuration (gristmill→Fortran+BLAS,
  pdaggerq-generated C++, SeQuant-generated code); we run tenferro. Same
  machine, identical input equations for every system (the same-equation
  rule applies regardless of backend). Report achieved GF/s alongside ms.
  Tiered targets: geomean >= 1.0x (parity) at t16 across the corpus =
  SUCCESS; geomean >= 1.1x = STRONG; running a problem size where the
  dense baselines exhaust memory (via Axis 5 storage) = HEADLINE.
  (Blocked on codegen, M3.)
- **Axis 4 — generality.** Drop-in problem: import a foreign-syntax
  equation set, produce verified competitive code, zero heuristic changes,
  within one day. (M4.)
- **Axis 5 — symmetry storage.** On problems with declared antisymmetry:
  >=2x peak-memory reduction with <10% wall-time penalty, oracle passing on
  symmetry-respecting inputs. (M5.)

## Baselines

Each baseline compiler is compared in two modes where feasible:
**(A) same-backend** — its optimized equations imported as candidates and
run through OUR stack on tenferro (diagnostic, Axis 2); **(B) native
system** — its own emitted code and backend, same machine (Axis 3; the
comparison that counts).

1. gristmill — A: done (`tools/gristmill_baseline.py`, same-equation).
   B: planned (M4).
2. pdaggerq — A + B (M4). Its pq_graph optimizer and C++ codegen make it a
   full-pipeline baseline.
3. SeQuant — A + B (M4). Factorization + libparenth ordering + codegen;
   strongest symbolic optimizer of the three.
4. TCE/OpMin — literature FLOP/memory numbers only; their regime
   (out-of-core) does not overlap our dense in-core envelope. Say so; do
   not fake-compare.

## Milestones

- M1: calibration suite + cost model v2 (rank/stride-aware slow path) +
  model-vs-measured CI gate. Closes F-004.
- M2: budgeted hybrid search + measured final acceptance; same-backend
  diagnostic green on the full corpus incl. ccsd_t. Closes F-005, F-007.
- M3: codegen to standalone Rust for all 7 problems.
- M4: second machine + pdaggerq/SeQuant imports + drop-in test. Axes 3, 4.
- M5: blockwise symmetry storage. Axis 5.
