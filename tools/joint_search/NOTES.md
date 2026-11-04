# Stage 1 results (joint-search prototype)

Branch: joint-search-proto. All numbers are model-side FLOP counts under the
truthmill convention (validated exactly against the exam's recorded
log10_flops for adc2_gristmill: 8.83695030414342). NO emitted candidate has
passed the numerical oracle yet — Stage 2 gate pending (needs worktree cargo
build). All conclusions provisional until then.

## Validations

- Parser reproduces known structure: ADC2 singles residual = 11 terms
  (matches F-007); nested LinComb-in-product distribution works
  (V11 = T3·(V0+V1+V2) etc.).
- Canonicalization (factor perms × domain/free-respecting label bijections)
  detects real sharing: ADC2 singles 4 shared subterms (fan-out ≤3), doubles
  4; CCSD 63-term def: 326 subterms, 142 sigs, 69 shared (fan-out ≤ 30).
- Term-level permutation-orbit grouping: ADC2 doubles 8 terms → 4 groups
  (matches the pair structure gristmill exploits).

## Cost comparisons (16,64), FLOPs, truthmill convention

| problem | gristmill | proto v1 (no sym) | proto v1.1 (+sym quotient) |
|---|---|---|---|
| ADC2 total | 686,989,824 | 689,154,048 (+0.3%) | — |
| CCSD total | 21,421,967,361 | 35,467,117,570 (+66%) | 25,488,222,272 (+19%) |

## Runtime

- ADC2: <0.1 s. CCSD 63-term definition: 0.44 s total (analyze + ILP) —
  compare F-005's kernel enumeration >400 s without completing.

## Diagnosed gaps (mechanisms)

1. **Sum-orbits (ADC2, −0.3%):** gristmill shares *sums* across permutation
   orbits (its V13 = A+B reused as V29, V30 = perm(V13)). Our term groups
   share evaluations but not partial sums. Exposing example:
   benches/candidates/adc2_gristmill.json values 9–13, 29–32.
2. **Nesting (CCSD, residual +19% after sym):** gristmill builds
   intermediates from intermediates; our sharing layer is one level.
   Fix: fixpoint — feed chosen intermediates' definitions back as
   factorizable structure.
3. **Sign tracking for symmetric factors:** canon_sig with SYMS ignores
   generator signs for matching; emission must fold orbit signs into
   contribution coeffs. Not yet implemented — emitted candidates so far use
   the non-symmetry solve (sign-safe by construction).

## Current gates

- [x] a (round-trip/parse): parser validated against known term counts.
- [~] b (ADC2 ≥ gristmill FLOPs): −0.3% behind — fails narrowly, mechanism
  understood (sum-orbits).
- [x] c (CCSD query < 10 s): 0.44 s.
- [ ] d (oracle): NOT RUN. Required before any further claim.

## Next

- Stage 2: worktree cargo build (quiet machine only), `harness check` on
  emitted candidates, v2 wall-model pricing of finalists.
- v1.1: fixpoint/nested sharing; sum-orbit detection; sign-correct
  symmetry emission.
