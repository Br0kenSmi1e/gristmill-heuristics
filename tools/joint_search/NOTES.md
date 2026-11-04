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

## Global multi-level ILP + sum intermediates (cost-space results)

Commands: `proto.py solveg <problem> 0=16,1=64 [sym]`. Cost-space only
(no emission yet for the global arm; sym-quotiented numbers are a
relaxation until sign-correct emission lands).

Complexity at 63-term scale: |U|=146, ~100 shared candidates, <1000 vars,
<910 constraints, build 0.3 s, solve instant. NO BLOWUP.

FLOP trajectory for CCSD doubles (63 terms), vs gristmill 21.42G:
- level-1 ILP, no sym: 35.47G (emitted, oracle-pass)
- global product-only, no sym: 35.11G  <- depth-greediness exonerated
- global product, sym: 25.20G
- global product + cofactor-family sums, sym: 23.02G  <- 7.5% above gristmill

Totals (all outputs, sym+sums): mp3 11.29G vs 17.18G (0.66x, win);
ccd 16.75G vs 16.63G (1.007x, par); ccsd 23.31G vs 21.42G (1.088x);
adc2 681.8M vs 687.0M (0.9925x).

Residual CCSD gap candidates: cross-cofactor sharing of one sum (gristmill
V10 pattern: same sum feeding multiple distinct cofactors), rests containing
intermediates (v2 restricts rests to plain DP), sums-of-sums.

Mechanism notes: group candidates = cofactor families keyed by (cofactor sig,
canonical rest-carried set); partial keep supported (cofactor may carry out
labels); singleton cofactors allowed (formation-free).

## Cross-cofactor sum sharing (global arm v3)

Motivated by direct measurement of gristmill's CCSD candidate: 7 sum
intermediates with 2-4 distinct einsum parents (cross-cofactor reuse).
Added s_sigma (shared sum, linkage-only) / h_gamma (per-family combine) /
r_{gamma,t} (per-member rest+add) variables — all linear.

CCSD doubles (63 terms): 22.49G vs gristmill 21.42G => 5.0% residual.
Trajectory: 35.47 (L1) -> 35.11 (global) -> 25.20 (+sym) -> 23.02 (+families)
-> 22.49 (+cross-cofactor). Totals: mp3 11.28G (0.657x win), ccd 16.74G
(1.007x par), ccsd 22.74G (1.061x), adc2 681.8M (0.993x).

Complexity: 1373 vars / 1362 cons / 0.4 s build at 63 terms — no blowup.

Residual gap candidates (unmeasured): rests containing intermediates (rests
are plain-DP only), sums of sums, term-permutation orbits not yet merged
into the global arm.

## Global arm v4: rests may contain intermediates => gap closed

Rest covers get their own block choices (sub-ILP structure, still linear).
CCSD doubles trajectory: 35.47 -> 35.11 (global products) -> 25.20 (+sym)
-> 23.02 (+families) -> 22.49 (+cross-cofactor sums) -> 21.20G
(+intermediates in rests) => 0.990x gristmill (21.42G).

Totals (sym+sums, cost-space): mp3 11.28G (0.657x), ccd 16.74G (1.007x),
ccsd 21.45G (1.001x), adc2 681.7M (0.992x). At or below gristmill everywhere.

Sanity control: no-sym (fully legal) CCSD doubles = 27.16G — gristmill
(21.42G) beats even our sym-relaxed space without it, so the remaining
structure was genuine, not a sign artifact. Sign-correct sym emission still
owed.

Complexity: max 1872 vars / 2375 cons / 0.5 s build at 63 terms.

## Global arm v5: joint cross-output solve + term permutation orbits

The user's argument verified empirically: a global optimizer over a complete
space must not sit above a greedy loop; each residual gap was a missing node
class. Final two classes added: term-permutation orbits (evaluate one member,
permute free — gristmill's adc2/ccd pair structure) and cross-output sharing
(one ILP over all definitions; gristmill shares ~80M between CCSD outputs).

Joint optima (cost-space, ILP-exact, status=0) vs gristmill:
- sym ON (relaxation — sign-legal emission owed):
  mp3 11.28G (0.657x), ccd 14.60G (0.878x), ccsd 16.40G (0.766x),
  adc2 514.0M (0.748x)
- sym OFF (fully legal space):
  mp3 11.28G (0.657x), ccd 20.81G (1.252x), ccsd 22.69G (1.059x),
  adc2 514.0M (0.748x)
  (mp3/adc2 have no declared symmetries — identical numbers validate the
  symmetry layer's consistency.)

Remaining legal-space gap (ccd 1.25x, ccsd 1.06x) = legal antisymmetry
exploitation, i.e. sign-tracking canonicalization + emission. Gristmill does
this; we don't yet.

Complexity ceiling so far: 1872 vars / 2375 cons / <1 s. No blowup anywhere.

Open classes (small, named): sums-of-sums; unsummed-precursor sharing;
lincomb micro-accounting (~1M on mp3-scale numbers).

## Global-arm emitter + full verification battery (stage: verify)

Global-arm emitter implemented: recursive formation emission (sigs by
increasing factor count — blocks are strict subsets, so acyclic), term
covers, family groups, perm-members, final lincombs. Battery:
`solvegemit <problem> 0=16,1=64 <out.json>` (partial_keep=False for
emission) + `harness check` at (3,5).

VERIFIED (emitted + truthmill validate + numerical oracle, worst rel
<= 5.2e-16, ILP objective == emitted price within lincomb accounting):
- mp3 11.283G (0.657x gristmill) — global-arm win, verified
- adc2 689.2M (1.003x) — par
- ccd 23.71G (1.426x) — behind
- ccsd 35.35G (1.65x) — behind

Cost-space-only (NOT yet realizable; ILP optima over the relaxed space):
- with sym quotient + partial-keep families + cross-cofactor sums:
  mp3 0.657x, adc2 0.748x, ccd 0.878x, ccsd 0.766x.
- legal (no-sym) cost-space: ccd 1.25x, ccsd 1.06x.

The realization gap decomposes into: (a) sign-correct symmetry emission
(sym numbers are a relaxation until then); (b) partial-keep family
emission (perm-members under partial keep need position-set-aware S
spaces; currently restricted to full-keep, which is sound and verified);
(c) both are engineering, not theory.

Bugs caught by the battery: entry-instance vs sig confusion in group
rests; MissingIndexDomain/UnusedIndexDomain in group steps; mispriced
partial-keep combine steps (137G step vs 22.7G objective — caught by the
priced-vs-objective self-check).

## Verification round 2: partial-keep families, perm direction fix

- numpy microtest settled the permute direction: einsum output=[sigma(p)]
  computes M[o] = R[o_{sigma(.)}], and the canon relation requires
  sigma = pi_t^{-1}. ALL corpus perms so far are self-inverse swaps, so
  earlier oracle passes could not distinguish pi from pi^{-1}; fixed in both
  permute sites. Latent bug class recorded (needs a cyclic-perm fixture).
- Families (partial keep) carry identity-perm members only; non-identity
  members of grouped reps get own plain covers, charged exactly in the ILP.
- Verified scoreboard (emitted + oracle @ (3,5)): mp3 0.657x win,
  adc2 1.003x par, ccd 1.39x, ccsd 1.51x.
- ILP objective vs emitted price agree within lincomb accounting on all four.

Remaining verified-gap lever: sign-aware symmetry exploitation
(canon_full now returns orbit signs; emission folding not yet wired).

## Sign-aware symmetry exploitation (closing the verified gap)

Sign conventions verified by numpy microtests: factor(instance) =
orbit-sign x factor(canonical); permutes use output=[pi_inv(p)]. Signs fold
into LinComb coefficients: whole-term signs for ungrouped terms, per-instance
signs (cofactor + rest) for family members.

Verified scoreboard, sym ON (oracle @ (3,5), truthmill::verify):
- mp3: 11.28G, 0.657x gristmill — PASS
- ccd: 16.90G, 1.017x gristmill — PASS (was 1.43x without sym)
- adc2: 689.1M, 1.003x — PASS
- ccsd: FAILS truthmill::verify on the doubles definition.

Minimal failing fixture for the open bug: terms {32,33,37,42} of the CCSD
doubles definition (extracted via parse_terms; subsets of terms as standalone
problems reproduce it). Pairs and the 3-term subsets pass; the 4-set fails.
Mechanism zone: nested shared intermediates under antisymmetry. All
bisection artifacts in /tmp/sub.json are reproducible via
tools/joint_search/proto.py.

Synthetic regression fixtures (pass): /tmp/antisym_test.json (perm pair +
antisymmetry), /tmp/antisym_test2.json (pair + shared subterm),
/tmp/ccsd_energy.json (3-term scalar).

## B-006 closed: two sign/frame errors (milestone)

Verified numerically with a harness-faithful numpy model
(tools/joint_search/diag_contrib.py — per-(tensor, domain-pattern) blocks,
stabilizer projection, generator-orbit propagation, mirroring
crates/harness/src/dense.rs):

1. **Formation orbit-sign drop**: `subset_sigs` discarded the orbit sign of
   a block instance inside a parent's canonical form. Mining happens in the
   original term label space and does not commute with canonical relabeling,
   so canonical parents can contain block instances with sign -1. Every
   materialization of the parent was then globally off by -1. Fixed by
   threading the sign through forms and folding the product into the
   formation tree's root einsum coeff.
2. **Perm-member double-emit/cancel**: family S-sums emitted a non-identity
   perm member's own rest with a pi-swapped placement AND the post-family
   section emitted the same member standalone (the ILP only charges the
   standalone). The pi-swapped member rest is exactly the negative of the
   rep's rest under antisymmetry, so the rep's term silently canceled.
   Fixed: S-sums skip non-identity members entirely.

Battery (trials/20261104-190100.jsonl), verified at oracle (3,5), FLOPs at
(16,64): mp3 0.6566x PASS | ccd 1.0163x | ccsd 1.1039x | adc2 1.0031x —
all four verify with sym on. New regression fixture
ccsd_doubles_36_37_50_51.json; the old B-006 fixture now guards (1).

Tooling added: tools/joint_search/term_bisect.py (ddmin over term subsets;
the file must not be named bisect.py — stdlib collision),
tools/joint_search/diag_contrib.py (per-contribution numeric bisector).

Remaining realization gaps vs the sym-relaxed ILP optima (0.766 ccsd /
0.878 ccd / 0.748 adc2): the emitted solves lock perm-membered reps out of
partial-keep families (C-010) and adc2's emitted price is ~7.3M above its
ILP objective (B-007 lincomb accounting — decisive for adc2's 1.0031x).

## B-007 + C-012: exact lincomb accounting, sum-orbits (milestone)

B-007 mechanism: the ILP never priced final lincombs (adc2: exactly
7,348,224 = 7,340,032 doubles + 8,192 singles), cross-output families were
charged once but emitted per output, S-adds counted standalone-bound perm
members. Fixed; obj - priced == const exactly on mp3/ccd/adc2 (ccsd
residual 8,448 FLOPs, accepted micro).

C-012 sum-orbits landed (emission + exact w_C cluster pricing): reps with
identical member (perm, coeff-ratio) signatures share one pre-sum; the
orbit rides it via free permutes. adc2: 689.09 -> 686.99M.

KEY RETRACTION: the v5-era adc2 514M (0.748x) "legal-space win" was
underpriced — v5 charged nothing for non-identity perm members of
family-covered reps (~173M on adc2). The honest adc2 optimum is
686,989,824 FLOPs == gristmill, an exact tie. The old sym-relaxed numbers
(ccd 0.878x, ccsd 0.766x) are likewise retracted: with honest pricing the
ILP optima sit ABOVE gristmill (ccd +1.6%, ccsd +10.3%). By the
space-completeness clause these residuals are missing structural classes
(C-013); adc2 needs a tie-break class (C-014).

Verified scoreboard (trials/20261104-192133.jsonl): mp3 0.6566 PASS |
ccd 1.0161 | ccsd 1.1031 | adc2 1.0000 (tie).

## C-015: canonicalization must mine orbits in canonical space (milestone)

The ccd +1.6% residual was NOT a missing node class — it was a
canonicalization defect: canon_full mined each factor's symmetry orbit in
ORIGINAL label space (lex order on arbitrary label ids), so equivalent
subterms with different label ids got different sigs and sharing was
missed (ccd formed the same T2.T1 rank-2 intermediate twice; one 134M
step had no gristmill counterpart). Fix: exact joint orbit min, made
fast by two observations: (1) the factor-permutation loop in
_canon_sig_core was dead work (the body is a sorted multiset); (2) order
statistics are per-element monotone, so for a fixed label renaming the
per-factor orbit-min in canonical space IS the exact joint orbit-min.
25-30x faster than the naive joint enumeration (mp3_sym 17.1s -> 0.59s;
budget gate restored).

Verified scoreboard (trials/20261104-194244.jsonl): mp3 0.6564 PASS |
ccd 0.99994 PASS (strict) | adc2 1.000000 (tie, C-014) | ccsd 1.0887
(C-013 residual 1.87G, under analysis).

## C-016 orbit-consistent families: ccsd beats gristmill (milestone)

The ccsd residual was NOT a missing node class either — it was a
restriction on family usage: families whose reps have non-identity perm
members were charged full standalone evals per member (post-B-006), so
the ILP rejected the big T3-cofactor families. Mechanism (verified by
lstsq term decomposition of gristmill's candidate): gristmill's
ccsd doubles covers 20 terms with ONE combine (T3 x (V58+V59)) plus three
free permutes — i.e. the whole orbit rides the family result. This is
sound whenever every covered rep shares one orbit signature
{C-016 consistency condition}. Implementation: (1) orbit_signature()
per group (identity members fold into c_eff; non-id members give
(perm, ratio) pairs); (2) family candidates are SPLIT into
signature classes so one inconsistent rep no longer poisons a family;
(3) consistent families emit one result + free permutes, and join the
C-012 cluster pre-sums; (4) priced exactly (no standalone `extra`,
lincomb units = 1 + |orbit|, cluster vars w_C <= h_gamma).

Verified scoreboard (trials/20261104-201217.jsonl), all oracle-green:
mp3 0.6564 PASS | ccd 0.9919 PASS | ccsd 0.9348 PASS | adc2 1.0000 tie.
C-013 closed (its "missing class" was C-016). Remaining goal item:
adc2's exact tie (C-014) — the space optimum == gristmill there, so a
class gristmill lacks is required.

## C-014 adc2 tie analysis (negative results, milestone critic pass)

adc2 is provably at its space optimum == gristmill (ILP exact, gap 0,
identical cost histograms). adc2 has NO declared symmetries. Measured
candidate classes for a strict beat, all zero/negative value on adc2:
- multi-operand fused steps (C-017): uniformly worse (13.4G vs 648M) —
  anti-class.
- value-level coincidences between emitted intermediates: none.
- C-009 unsummed-precursor: no chosen block recurs with a different keep
  set; the T1.T4 3-axis contractions (singles reps 9/10) would need a
  shared 2-axis precursor of size v^3o^3 (2.1G) >> 2x134M — dead.
- gristmill's adc2 uses only landed classes (families with sum operands,
  sum-orbits L13/L18 pair pattern, free permutes).

Critic-pass note: ccsd priced is 8.2K BELOW the ILP objective (the
accounting now slightly overcounts — likely cluster-presence edge);
cosmetic, flagged for the next B-007 follow-up. adc2 remains the only
goal blocker: mp3 0.6564 | ccd 0.9919 | ccsd 0.9348 | adc2 1.000000.
