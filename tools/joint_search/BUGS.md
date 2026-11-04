# BUGS — the bug registry (challenger arm)

One entry per bug: symptom, mechanism (verified BEFORE fix), exposing
fixture, fix, regression pointer. CLOSED entries kept for audit.

## B-001 permute direction (pi vs pi^{-1}) [CLOSED]
- symptom: wrong values whenever a non-self-inverse output permutation
  rides a shared evaluation.
- mechanism: VERIFIED by numpy microtest — einsum output=[sigma(p)]
  computes M[o] = R[o_{sigma(.)}], and the canonicalization relation
  requires sigma = pi^{-1}. Corpus perms are all self-inverse swaps, so
  the oracle could never catch this on the train problems.
- fix: both permute sites use pi_inv.
- regression: microtests.py::test_permute_direction; NOTE the corpus
  lacks a cyclic-perm fixture — fixture gap recorded (make one).

## B-002 partial-keep combine mispricing [CLOSED]
- symptom: emitted CCSD program contained a 137G-FLOP step against a
  22.7G ILP objective.
- mechanism: VERIFIED by the priced-vs-objective self-check — partial-keep
  family combine steps priced with a wrong label set.
- fix: partial-keep families carry identity-perm members only;
  non-identity members get own covers, charged in the ILP.
- regression: the self-check itself (gate b in LOOP.md).

## B-003 domain maps of group steps [CLOSED]
- symptom: MissingIndexDomain / UnusedIndexDomain validation errors.
- mechanism: S-sum and combine steps' `domains` covered the wrong label
  sets (too wide / too narrow).
- fix: domains = exactly the labels used in the step.
- regression: full battery.

## B-004 entry-instance vs sig confusion in group rests [CLOSED]
- symptom: when two entries of one term shared a canonical sig, the rest
  dropped both — phantom output labels.
- mechanism: grouped variants identified the factored entry by sig, not
  by instance index.
- fix: variants carry the entry's position in the subset.
- regression: full battery; mp3 (6 terms with repeated sigs) covers it.

## B-005 sign over-folding [CLOSED]
- symptom: ccd/ccsd failed truthmill::verify with sym on while passing
  with sym off.
- mechanism (three separate frame errors, all verified): (i) whole-term
  signs (tsigns) applied to reps that emit written singletons — singleton
  orientations must NOT be folded; (ii) rep's sign applied to group
  members — each member needs its own instance sign; (iii) plain-emitted
  members double-folded. All pinned by bisection + the antisymmetry
  fixtures.
- fix: per-entry orbit signs; rep coeff = product of its cover's block
  signs; member coeffs = tsigns[m]*tsigns[rep]*block_sign (ungrouped) or
  per-member (s_g, s_rest) (families).
- regression: fixtures/antisym_test.json, fixtures/antisym_test2.json,
  fixtures/ccsd_energy.json.

## B-006 CCSD doubles verify failure under sym [CLOSED]
- symptom: ccsd doubles failed truthmill::verify with sym on; passed with
  sym off. ccd/mp3/adc2 passed with sym on.
- mechanism: TWO frame errors, both verified numerically
  (tools/joint_search/diag_contrib.py, harness-faithful per-(tensor,
  domain-pattern) block model):
  (i) **formation orbit-sign drop**: `subset_sigs` used `canon_sig`,
  discarding the orbit sign of a sub-block instance inside a parent's
  canonical form. Orbit-mining does not commute with canonical
  relabeling, so a block can sit inside a canonical parent with sign -1
  (exposed: T1[1,0,4,5] inside the T1+3xT2 intermediate, mined to
  (0,1,4,5) with sign -1). Every materialization of such a sig was off
  by -1 everywhere it was used. Fix: `subset_sigs` yields the sign via
  `canon_full`; the formation folds the product of block signs into the
  root einsum coeff of the emitted tree (`emit_binary_tree(sign=...)`).
  Fixture: fixtures/ccsd_doubles_38_39_40_42_43.json (terms
  {38,39,40,42,43}).
  (ii) **non-identity perm member double-emit + cancellation**: the
  family S-sum loop emitted a perm member's own rest with a pi-swapped
  input placement, while the post-family section ALSO emitted the same
  member as a standalone plain evaluation (the ILP charges only the
  standalone). Under antisymmetry the pi-swapped member rest is the
  exact negative of the rep's rest, so the rep's term canceled out of
  the S-sum. Fix: family S-sums skip non-identity members; they are
  covered solely by the standalone path (which the ILP prices).
  Fixture: fixtures/ccsd_doubles_36_37_50_51.json (terms {36,37,50,51}).
- exposing fixtures: the two above (regenerate via make_fixtures.py;
  bisect via tools/joint_search/term_bisect.py).
- regression: full battery all-pass (trials/20261104-190100.jsonl);
  ccsd_global_sym verified 1.1039x.
- history: terms {32,33,37,42} failed before B-005; {38,39,40,42,43}
  isolated (i); after (i), {36,37,50,51} isolated (ii) via term_bisect
  ddmin over the 63-term doubles definition.

## B-007 lincomb micro-accounting [CLOSED]
- symptom: ILP objective vs emitted price differed by tiny amounts
  (mp3: 2 FLOPs; adc2: ~7M of 689M).
- mechanism: VERIFIED — the ILP never charged the final lincombs
  (adc2: 7,340,032 doubles + 8,192 singles = the exact delta), family
  S-sum/combine costs were charged once even for cross-output families
  (emitted per owning output), S-adds counted standalone-bound perm
  members, and r-covers carried an obsolete extra_rest multiplier.
- fix: exact accounting — final-lincomb units per contribution (z reps:
  1 + #non-identity members; families: 1 + #non-id per owning output),
  per-output S/combine pricing, identity members folded into rep coeffs
  (C-012 emission), extra_rest dropped. Post-fix: obj - priced = const
  exactly on mp3/ccd/adc2; ccsd residual 8,448 FLOPs (3.6e-5 relative;
  accepted as micro-residual, cause unknown).
- became goal-decisive: adc2's objective already beat gristmill; the
  honest accounting + C-012 closed the emitted gap to exactly 1.000000.
