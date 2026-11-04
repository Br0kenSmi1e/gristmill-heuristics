# Joint-search challenger: design + literature check

Prototype branch `joint-search-proto`. This document is self-contained to the
branch; **no ledger files** (FAILURES.md, IDEAS.md, RESEARCH.md, trials/) are
modified here. Filing as a proper I-entry is a later, deliberate mainline
decision.

Theoretical background: the structure-tensor network-factorization note
(`~/Documents/tensor-factorization-encoding.pdf`, v2). This doc is the
engineering plan: what we build, why we believe it can win, how it is gated.

## 1. What we build

A **challenger candidate generator** for the contested factorization/selector
slot, replacing single-move greedy search with solver-based joint optimization:

1. **ILP sharing layer** (HiGHS via `scipy.optimize.milp`, pure Python):
   binary variables over the term×subterm structure decide which subterms
   become shared intermediates and which terms route through them. Objective:
   total FLOPs (linear in the binaries — a node's label set fixes its cost
   monomial, so extents enter as constants). The F-018 Axis-1 cap
   (≤ 1.05× baseline FLOPs) is a linear constraint, enforced in-model.
2. **Exact DP parenthesizer** for each resulting term (subset DP, k ≤ ~6
   factors — microseconds, optimal).
3. Emit **candidate JSON** in the existing `benches/candidates/*.json` format.
   No changes to `crates/` — the harness stays untouched; the challenger is a
   pure producer of candidates.
4. Verification via the worktree-built harness: `harness check` (numerical
   oracle) + v2 wall-model pricing of finalists. All execution happens in
   this worktree only.

Why two levels: sharing decisions are the complicating variables; with them
fixed, parenthesization is exact and free. (See the note, §11.)

## 2. Why we believe it can win (mechanism-to-failure matching)

- **F-007 (verified greedy trap, ADC2):** the v2 model already ranks the
  gristmill schedule better than the selected one (105.1 vs 119.9 ms
  predicted; measurement agreed). The landscape is right; the deficit is
  *reachability*. A global optimizer (ILP) and non-greedy search address
  exactly reachability. If I-007's outcome says "(b) search problem", this
  challenger is the natural M2 investment it selects.
- **F-005 (enumeration explosion, CCSD 63-term):** the kernel enumerates all
  maximal bicliques (>400 s). An ILP never enumerates — LP bounds + conflict
  learning prune the same space. ~10⁴ binaries at CCSD scale is routine.
- **F-008 (chunking destroyed cross-chunk factorizations):** the ILP is
  monolithic over all terms — couplings are present by construction.

Falsifiable predictions: on the ADC2 singles residual (11 terms), the ILP+DP
arm reaches ≤ gristmill FLOP cost within minutes; on CCSD's 63-term
definition, the sharing query finishes within a 10 s budget where enumeration
exploded. Failure on ADC2 — small enough to be damning — kills the approach.

## 3. Fixtures and gates

- **Fixtures:** ADC2 singles residual (F-007's verified case,
  `benches/chemistry/adc2.json` + `benches/candidates/adc2_{selected,gristmill}.json`)
  and CCSD's 63-term definition (F-005).
- **Gate a (round-trip):** parse and re-emit `adc2_gristmill.json`;
  byte-level structural equality of the term/step structure.
- **Gate b (optimality):** ADC2 residual — match or beat gristmill FLOPs.
- **Gate c (budget):** CCSD 63-term sharing query < 10 s.
- **Gate d (correctness):** every emitted candidate passes `harness check`
  (numerical oracle) in this worktree; Axis-1 cap verified against the
  same-equation baseline.
- **Load discipline:** all runs record `uptime`; no timing runs while the
  mainline agent benchmarks. No protected-set tuning: develop on
  {mp3, ccd, ccsd, adc2-train}; never touch the protected problems.

## 4. Literature check (2026-11-03, web)

Required before any I-entry; recorded here.

**Sum-product / tensor-program optimization with e-graphs (the closest line):**
- SPORES (Wang, Hutchison, Leang, Howe, Suciu; VLDB 2020;
  arXiv:2002.07951): sum-product optimization via relational equality
  saturation; recovers all hand-coded SystemML optimizations. Direct
  precedent for distributivity-driven CSE/factoring via e-graphs.
- Tensat / "Equality Saturation for Tensor Graph Superoptimization" (Yang,
  Phothilimthana, Wang, et al.; MLSys 2021): TASO re-implemented with egg;
  tensor graph superoptimization. TASO (PLDI 2019): substitution-based
  graph search. Glenside (MAPS 2021): pure tensor IR for egg rewriting.
- Known e-graph explosion on AC+distributivity is the standing risk; our
  mitigation: AC by canonicalization (n-ary sorted normal forms), directed
  distributivity, size caps — and e-graphs are *demoted* behind ILP/local
  search in our stack.

**Matrix-multiplication algorithm discovery (AlphaTensor line):**
- Fawzi et al., AlphaTensor (Nature 2022); AlphaEvolve (2025).
- Heule–Kauers–Seidl, "Local Search for Fast Matrix Multiplication" (SAT
  2019, arXiv:1903.11391) and "New ways to multiply 3×3-matrices"
  (13,000+ inequivalent 23-multiplication schemes via SAT).
- Kauers–Moosbauer, "Flip Graphs for Matrix Multiplication" (ISSAC 2023,
  arXiv:2212.01175): stochastic local search on decompositions beat the
  RL approach — the precedent for our flip-style arm. 2025 follow-up
  (arXiv:2502.04514) adds symmetry reduction of the search space — direct
  precedent for our declared-symmetry quotienting.

**Contraction ordering (the pure ⊗ layer):**
- Markov–Shi (2005): contraction ≍ treewidth duality.
- Dumitrescu et al. (PLOS One 2018): treewidth solvers practical for
  tensor-network simulation.
- NP-hardness sharpened (arXiv:2310.06140). opt_einsum / cotengra are the
  mature heuristic tooling.

**Quantum-chemistry domain tradition (must cite, our problem's home):**
- Tensor Contraction Engine (Hirata 2003; Hartono et al., JPCA 2009):
  operation minimization via commutativity/associativity/distributivity +
  CSE on coupled-cluster equations.
- Hartono et al., "Identifying Cost-Effective Common Subexpressions…"
  (ICCS 2006): two-step single-term-optimization/CSE *because* their
  combined space is exponential — already cited by I-006. Our joint
  formulation is precisely the combined space they declined.
- Datta–Gauss non-antisymmetric TCE (2013) for spin-adapted CC.

**Novelty boundary (no claim made):** the unrestricted combination —
structure-tensor network factorization + measured multi-regime backend cost
functor + joint sharing/ordering with solver certificates, on multi-output
symmetry-declared QC equations — is not found in the above. Each component
exists; the joint, cost-functor-driven formulation appears unclaimed.
Re-check before any I-entry moves past `priced`.

## 5. Format notes (from inspection of this branch)

- Problem JSON: `{"sizes": {space_id: extent}, "symmetries": {tensor: [...]},
  "reference": {"values": [terms]}}`; a term is
  `{"Einsum": {"coeff": [num, den], "code": {"inputs": [[labels]...],
  "output": [labels], "domains": {label: space_id}}, "args": [{"Tensor": id}...]}}`.
  Signed rational coefficients exist (e.g. `[-4, 1]`) — the signed branches
  of the encoding are native to the format.
- Candidate JSON: `{"values": [steps]}`; steps are `Einsum` or `LinComb`;
  args reference `{"Tensor": id}` (inputs) or `{"Value": k}` (step-k
  intermediate) — the schedule DAG is already the network of §4.4 of the
  note.
- Declared sizes in `adc2.json` are nominal (1000/5000); evaluation sizes
  come from `--sizes` at run time.

## 6. Stage plan and status

- [x] Stage 0: worktree + literature check + this design doc.
- [ ] Stage 1: Python prototype (parse → subterm structure → HiGHS ILP →
      DP parenthesizer → emit candidate JSON). Gates a–c.
- [ ] Stage 2: worktree cargo build (quiet machine only); oracle + v2
      wall-model verification of emitted candidates. Gate d.
- [ ] Stage 3 (optional arm): flip-style local search over the joint move
      set, measured against the ILP dual bound.
- [ ] Report back to mainline; I-entry + tournament decision with the user.

Status: Stage 0 complete at branch HEAD.
