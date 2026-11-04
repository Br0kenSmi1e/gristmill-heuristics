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

## B-006 CCSD doubles verify failure under sym [OPEN]
- symptom: ccsd doubles fails truthmill::verify with sym on; passes with
  sym off. ccd/mp3/adc2 pass with sym on.
- mechanism: NOT verified. History: terms {32,33,37,42} failed before the
  B-005 sign fixes; that set passes now. Current reproducer (post-fix):
  terms {38,39,40,42,43} (five T1-containing terms with mixed
  T2/T3-cofactor nesting; every proper subset passes). Zone: nested shared
  intermediates under antisymmetry with perm-group interaction.
- exposing fixture: fixtures/ccsd_doubles_38_39_40_42_43.json
  (generate via make_fixtures.py). NOTE: fixture membership drifts as
  fixes land — regenerate and re-bisect before trusting it.
- next: per-contribution numeric bisector already exists (compare each
  LinComb arg against the true term).

## B-007 lincomb micro-accounting [OPEN, minor]
- symptom: ILP objective vs emitted price differ by tiny amounts
  (mp3: 2 FLOPs; adc2: ~7M of 689M).
- mechanism: final-lincomb arg counting is not in the ILP objective;
  off-by-one add conventions per group.
- not a correctness issue; fix when the wall-time functor lands (the v3
  pricer will reprice everything anyway).
