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
  privilege — it is a component, not an identity. The a8bf146 exam falsified
  the assumption that the incumbent is Axis-1 green. Emission is now capped
  with baseline fallback (F-018); further displacement still requires a green
  exam. I-001 and the structural proposer
  are currently NOT EVALUABLE, not silently adopted or retired (F-025).

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

## I-005 bounded lookahead over alternative parenthesizations [retired(evidence)]
- hypothesis: retaining a bounded set of parenthesize→factor / parenthesize→
  parenthesize paths escapes the F-007 greedy local optimum without requiring
  gristmill's full TRAV search.
- wins IF: a two-move path improves the current model objective where every
  constituent first move is rejected; the ADC2 singles residual is the
  exposing fixture, not a tuning set.
- price: top-8 first moves by symbolic FLOP delta and top-8 second moves per
  first move; <=64 two-move terminal scores plus <=8 nested one-move action
  enumerations per round, under 10x one-step scoring work by construction.
  Actual compile-time ratio remains an adoption gate.
- cheapest test: implement bounded two-move scoring behind an experimental
  selector flag, then compare one-step vs two-step on {ccd, ccsd} under the
  model. Measure the best candidate on train only; evaluate ADC2/EOM once at
  the milestone.
- adopt IF: Axis 1 stays <=1.05x same-equation gristmill on every problem,
  train geomean measured wall time improves >5%, no train problem regresses
  >2%, and selector compile time is <10x one-step greedy. Otherwise retire
  with the numbers.
- addresses: F-007.
- result: retired after answering F-010 with explicit
  `H_BICLIQUE_MAX_TERMS=1000`, measured train A/B, and the ADC2 fixture.
  Valid CCSD model cost was 1.1425 s one-step vs 1.1809 s lookahead, but
  measured wall time favored lookahead 794.7 vs 737.3 ms (1.078x). CCD was
  451.6 vs 452.8 ms (0.997x); the TWO-problem geomean was 1.037x. MP3 and EOM
  arms were not run, so the full train-geomean and EOM clauses are unevaluated
  (F-020). Retirement is instead dispositive on Axis 1: challenger ratios were
  1.265x CCD and 1.600x CCSD, above the required 1.05x. ADC2 gave only 0.049%
  terminal model improvement and did not meet wins-IF. Apparent gains on
  chunked controls were separately inadmissible (F-008).
- revival rule: fired by the model v2 landing (0b44e61). Resolved WITHOUT a
  tournament: the disqualifying clause was measured Axis-1 flop ratios
  (1.265x CCD, 1.600x CCSD), which are invariant under cost-model changes,
  so retirement stands.

## I-006 budgeted maximal-biclique enumeration [proposed]
- hypothesis: a gristmill `greedy_cutoff`-style depth budget inside the
  gristmill-rewrites Bron–Kerbosch enumeration retains the profitable
  bicliques while avoiding the F-005 exponential query tail.
- wins IF: bounded queries preserve >=99% of full-query FLOP savings on
  train definitions whose full queries complete, while large train defs
  finish within a 10 s/query hard budget.
- cheapest test: add an upstream enumeration-budget API and compare bounded
  vs full candidate sets on {ccd, ccsd}; do not execute ccsd_t. Evaluate its
  static query/candidate count only at a later resource-safe milestone.
- adopt IF: Axis 1 stays green, no train schedule regresses >1% under the
  time objective, every bounded query meets its budget, and end-to-end
  compile time is <10x the split-kernel control. Otherwise retire with numbers.
- addresses: F-005; follows the verified failure of the structural fallback
  recorded there.
- origin / prior art: gristmill `optimize(..., contr_strat=ContrStrat.TRAV)`
  retains traversed contraction evaluations for subsequent sum optimization;
  local source inspected at gristmill/optimize.py:3025-3037 and public repo
  https://github.com/DrudgeCAS/gristmill. Hartono et al., "Identifying
  Cost-Effective Common Subexpressions to Reduce Operation Count in Tensor
  Contraction Evaluations", ICCS 2006, DOI 10.1007/11758501_39, describes a
  two-step single-term-optimization/CSE approach because their combined space
  is exponential. Web prior-art check: 2026-10-29; no novelty claim made.

## I-007 space-vs-search discrimination experiment [tested]- question (F-007 reframed): the v2-priced greedy selector finds nothing
  better than baseline on 5/6 problems. Is that because (a) the rewrite
  space CONTAINS nothing better, (b) the space contains better schedules
  greedy cannot reach, or (c) pricing is still wrong?
- hypothesis: enumerating the full kernel candidate space on train problems
  and model-pricing ALL of it (not just the greedy path) shows whether
  sub-baseline candidates exist at all.
- cheapest test (model-priced, no new algorithm): dump every enumerated
  candidate's v2 price on {ccd, ccsd, adc2}; compare min-price vs baseline
  price vs greedy's reached price. Interpretation: min(space) >= baseline
  => (a) space problem; min(space) < baseline but greedy misses it => (b)
  search problem; prices invert vs measured anchors => (c) model problem.
- adopt IF: n/a — this is a measurement, not a component. Its outcome
  SELECTS which of I-001 (search depth), structural-proposer improvements
  (space), or model v3 (pricing) is the M2 algorithm investment.
- prerequisite: F-028 (selector nondeterminism) must be closed first —
  space enumeration is meaningless if the emitter is nondeterministic.
- addresses: F-007; gates the M2 algorithm choice.
- RESULT (trials/i007.jsonl, instrument `harness space-dump`): adc2 is
  SPACE-bound and measurement-corroborated (min 0.1185 s vs baseline
  0.1049 s; the space min IS the v1-era selected's price — the search
  found it). ccd and ccsd are confounded by leg (c): the model prices
  in-space states 17-51% above baseline while those states measure at
  parity or better (ccd 0.8313-priced vs 0.996x measured; ccsd v1-era
  selected 1.058-priced vs 0.947x t1 / 1.201x t16 measured) — pricing
  error exceeds the space gap, so no space/search conclusion is sound
  there. ccsd caveat: budgeted best-first (80 expansions, 6498 states)
  missed the known reachable v1-era selected state — instrument budget
  limitation recorded. M2 SELECTION: model v3 (F-016 gap classes) FIRST;
  I-001 (search depth) unsupported on all three; adc2 additionally shows
  the kernel space itself tops out above baseline. Bonus finding: the
  selector's FLOP top-8 prefilter causally blinds it (ccd prefilter-free
  greedy reaches 0.831 s where the selector converged at step 0) — a
  selector bug independent of M2.

## I-008 structural proposer as factorization challenger [priced]
- origin: written in response to F-034 (critic: the contested-slot
  challenger roster was never written, so the incumbent was re-seated by
  procedure at two milestones). The proposer is implemented
  (`select --engine structural`, `crates/harness/src/propose.rs`;
  benches/candidates/*_struct.json are its v1-era artifacts) and carries
  the F-004 exposing case and F-005's CCSD failure.
- hypothesis: definition-level structural factorization (no kernel
  biclique enumeration) produces schedules competitive with the incumbent
  kernel selector inside the kernel's envelope, and remains defined above
  it (the F-005 motivation).
- cheapest test (this milestone's tournament, mechanical): regenerate the
  struct candidates under the CURRENT engine (model v2 + F-018
  cap/fallback) on {ccd, ccsd, ccsd_lambda, eom_ccsd, adc2} (mp3: no
  structural action space exists — not evaluable there; ccsd_t: execution
  blocked by F-017); compare flop ratios and t1/t16 measured wall time vs
  the incumbent's exam-cd921bd numbers, 10/3, load-guarded.
- adopt IF (the contested slot's displacement rule, applied mechanically):
  Axis 1 <= 1.05x on every corpus problem the proposer emits for, AND
  Axis 2 geomean over the evaluable problems beats the incumbent's
  exam-cd921bd selected numbers with no problem regressing >2%. Otherwise
  the incumbent stays and the numbers are recorded here.
- addresses: F-034; contested slot: factorization engine.
