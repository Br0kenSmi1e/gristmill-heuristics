# IDEAS — the algorithm ledger

Each entry: hypothesis, mechanism prediction (a falsifiable "wins IF"
condition), cheapest decisive test, pre-registered adoption rule, status.
Status machine: proposed -> priced -> tested -> adopted | retired(evidence).

Revival rule: any retired idea gets re-priced when the cost model major
version changes (a idea retired under a wrong model may deserve a retrial).

## Contested slots (standing)

Some compiler components are permanently open to challenge at tournaments:

- **Factorization engine.** Incumbent: gristmill-rewrites kernel (works,
  verify-gated). Challengers: I-001 (ILP selection), the structural
  proposer, anything proposed later. Displacement rule: challenger must
  keep Axis 1 green (<=1.05x gristmill FLOPs on every corpus problem) AND
  beat the incumbent on Axis 2 numbers. The incumbent has no other
  privilege — it is a component, not an identity.

## I-001 ILP selection over enumerated intermediate candidates [proposed]
- hypothesis: gristmill-greedy round-by-round selection leaves wall time on
  the table; exact selection (CP-SAT) over the same candidate space closes
  the F-007 gap on eom/adc2.
- wins IF: greedy's per-round choices conflict on a substantial fraction of
  terms (measurable: count candidate conflicts on ccd/ccsd/eom before any
  optimization).
- cheapest test: enumerate maximal bicliques (bounded), CP-SAT select under
  the TIME objective, compare vs greedy selector on ccd/ccsd/ccsd_lambda —
  model-priced first, measured top-2.
- adopt IF: >10% under the time objective on >=2 corpus problems AND
  compile time <10x greedy. Otherwise retire with the numbers.
- origin: gristmill-autoresearch DESIGN.md Stage 2 (designed, never run).
- addresses: F-007.

## I-002 joint contraction-tree + layout DP [proposed]
- hypothesis: two-phase optimization (tree first, layout after) is
  suboptimal because tree choice changes layout cost and vice versa.
- wins IF: best-tree layout cost differs materially from
  best-(tree,layout)-jointly cost on corpus ops (model-priced diff).
- cheapest test: exact DP over factor subsets (terms <= 8 factors) with
  output layout in the DP state, on the 20 most expensive ops of ccsd.
- adopt IF: >5% geomean wall-time improvement on train corpus at <10x
  search cost. Otherwise retire.

## I-003 symmetry-aware blockwise scheduling [proposed]
- hypothesis: antisymmetry-aware blockwise storage cuts peak memory enough
  to matter at QC sizes, without GEMM fragmentation eating the win.
- wins IF: fragmentation keeps per-block GEMM sizes above the calibrated
  efficiency knee (needs M1 calibration data first).
- cheapest test: static analysis of ccsd_t2 block sizes at (32,96) under
  blockwise-compressed scheme; price with model v2.
- adopt IF (Axis 5): >=2x peak-memory reduction with <10% wall-time
  penalty on problems with declared antisymmetry. (M5.)
- addresses: the machine-envelope constraint (SPEC regime note) — the
  headline-win path on Axis 3.

## I-004 tenferro upstream fixes as compiler moves [proposed]
- hypothesis: where the ceiling is backend kernel quality (e.g. 0.6 GF/s
  rank-6 ops), a backend PR dominates any scheduling workaround.
- wins IF: roofline check shows achieved GF/s far below the calibrated
  envelope for an op class NO layout choice fixes.
- cheapest test: roofline report per emitted schedule (part of M1 harness).
- adopt IF: an op class is >5x below envelope under all layouts; then file
  upstream issue with the calibration data attached.
