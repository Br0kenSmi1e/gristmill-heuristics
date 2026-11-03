# FAILURES — the failure registry

One entry per failure: symptom, mechanism (verified BEFORE fix), exposing
fixture, fix, regression pointer. CLOSED entries are kept for audit
(including retractions — a retraction is a finding, not an embarrassment).

## Open queue (work items live here; read this first)

- **F-005** kernel enumeration explodes on large defs — I-006
- **F-007** selector behind gristmill on ADC2/EOM — ADC2 mechanism verified;
  under model v2 the selector beats baseline only on mp3 (exam-cd921bd)
- **F-016** model error untracked on held-out — calibration owned by M1;
  new v2 gap classes (k-heavy low-rank-out, lincomb rank-6) logged
- **F-017** CCSD(T) memory: HARD blocker, no full-size reruns
- **F-035** backend n-ary execution planning nondeterministic per process
  (F-028's mechanism on the execution side; wall-time impact unmeasured)
- **F-036** migrate calibration + exams to the release binary profile
  (F-033: debug does not preserve ratios — the mp3 win is 1.19x debug,
  1.02-1.04x release t1)
- **F-027..F-034** milestone critic challenges — ALL ANSWERED (F-028/033/
  027 closed with new evidence: deterministic model planner, release-ratio
  falsification, mp3 replication; F-029/030/031/032/034 closed on records;
  F-035/F-036 opened in response)
  (see the third "Critic challenges" section)

Entries below are in rough filing order, not priority order; the two
"Critic challenges" sections are milestone-review records.

Seeded retroactively from the prequel
study (archive/RESEARCH-v1.md, tag `hl-study-final`).

## F-001 per-use argmax layout starves multi-use tensors [CLOSED]
- symptom: layout v0 regressed CCSD to 1.45x SLOWER than naive
- mechanism: each tensor's layout chosen by its single most demanding use;
  multi-use intermediates landed in layouts bad for their aggregate traffic
- fix: uniform storage-entity cost model (v2), consumer-aware (M2 era)
- regression: layout v2 path in `harness time --layout model`

## F-002 greedy n-ary splitting regresses CCSD to 0.53x [CLOSED]
- mechanism: tenferro plans small n-ary einsums well internally; forced
  binary splitting multiplied dispatch and permute overhead
- fix: split only when memory forces it (factmem transform)
- regression: transform=`cse`/`factmem` split heuristics

## F-003 oracle ignored cross-block symmetry relations [CLOSED]
- symptom: the (correct) imported gristmill CCD candidate FAILED the
  numerical oracle while passing truthmill::verify
- mechanism: input blocks were projected per-block; cross-block symmetry
  relations the candidate legitimately exploits were not present in inputs
- fix: one projected representative per symmetry component
- regression: `harness check` on gristmill candidates

## F-004 cost model blind to high-rank slow path [CLOSED — model v2 priced the reduction regime; CI gate in place]
- symptom: ccsd_t candidate predicted 0.41 s, measured 8.1 s (20x error);
  model-only acceptance shipped a schedule SLOWER than naive
- mechanism (v2, VERIFIED by the M1.1 calibration grid,
  profiles/calibrate-v1.json, two guard-compliant sessions agreeing within
  3% on all >=100 ms cases): the hole is the backend's REDUCTION path, not
  operand count and not index rank per se. Output-rank 0/1 contractions run
  at 0.02–0.09 GF/s t1 — including a plain rank-1 dot product "a,a->" at
  0.088 GF/s — 250–3000x below clean GEMM (57.7 GF/s) and ~250x below the
  modeled slow rate (22.5). This path has NO t1->t16 thread scaling;
  rank-1-output ops (bin_r6_out1) run 6–15x SLOWER at t16 (unstable:
  362 vs 945 ms across sessions). Rank-6-output formation is a third,
  intermediate class (0.4–6 GF/s). Controls: low-rank n-ary chains reach
  455–575 GF/s (operand count irrelevant); chaining a ternary reduction
  into binaries does NOT rescue (each binary reduction inherits the slow
  path). The grid reproduces the in-situ miss exactly: tern_r6_scalar at
  (4,8) = 3.24 ms vs ccsd_t_struct op#45's reported 3.32 ms, model 0.03 ms
  (108x).
- original mechanism note (superseded in kind, not in direction): binary
  clean/slow GEMM classification too coarse; rank-6 intermediate-forming
  ops run at ~0.6 GF/s.
- fix (landed 0b44e61): cost model v2. Steps with output rank 0/1 price at
  the measured reduction rates (0.09 aligned / 0.04 strided / 0.05 rank-1-out
  GF/s t1, thread-flat; 0.006 GF/s t16 rank-1-out, unstable regime priced
  pessimistically); pure batched multiplies (m=n=1) are bandwidth-priced;
  rank>=5 steps use bw_hi (0.75/6.5 GB/s t1/t16) plus a measured 1.3 ms
  per-step fixed overhead; GEMM anchors recalibrated from
  profiles/calibrate-v1.json (57.7/255.3 clean, 23.0/153.0 slow).
- regression: `harness model-ci` (pure model evaluation against the
  committed calibration grid; no timing) + cargo test
  `model_v2_within_2x_of_calibration_grid`. Three exceptions documented
  with reasons in MODEL_CI_EXCEPTIONS (bin_r6_out1 t16 unstable;
  tern_lowrank_control small-m dispatch; quad_r6_scalar t16 3.9x session
  variance).
- acceptance: MET. ccsd_t_struct at (4,8): worst einsum ratio was 110x,
  now within [0.40x, 1.41x]. Residual gaps (rank-6 lincomb 10-17x,
  k-heavy low-rank-out contractions 26-500x) are newly surfaced PRE-EXISTING
  v1 holes, not v2 regressions — tracked in F-016.
- side effect (recorded, trials/model-v2.jsonl): v2 repricing converges the
  CCD selector at step 0 -> Axis-1 baseline fallback. The archived
  v1-era selected schedule scores worse than baseline under v2 (0.831 vs
  0.708 s predicted); exam-grade CCD measurement was ~parity (0.996x t1),
  so the measured cost of the fallback is ~zero. Whether v2 misranks these
  is milestone evidence, not tuned against.
- exposing case: ccsd_t structural-proposer candidate (trials-v1/struct_all.jsonl);
  ADC2 selected-vs-gristmill gap underprediction (F-016); calibrate cases
  tern_r6_scalar / bin_r6_out1 / dot_r2_scalar.

## F-005 kernel biclique enumeration explodes on large defs [OPEN]
- symptom: ccsd_t def0 query >400 s (kernel enumerates ALL maximal
  bicliques, no budget); selector skips defs >20 terms
- planned fix (M2): enumeration cutoff (greedy_cutoff analogue); see I-006.
  The current structural proposer is NOT a sufficient fallback: on CCSD's
  63-term train definition, wall-model hybrid accepted zero moves and left
  log10 FLOPs 13.875 (split kernel 12.913); FLOP-priced hybrid's best proposal
  failed exact equivalence. No CCSD(T) execution was performed.
- fix candidates: I-006 (bounded kernel enumeration).
- exposing case: ccsd_t via `harness search` (kernel path); CCSD 63-term def
  is the safe train proxy for fallback experiments.

## F-006 benchmark contamination by concurrent load [CLOSED — runner-level guard built]
- symptom: M3-era batches measured under load avg 24 (~20 julia jobs);
  absolute timings invalid, required full re-measurement
- mechanism: shared machine, no guard; wall-time comparisons silently wrong
- fix: structural. Every timing command (`time`, `profile`, `cost-check`,
  `report`, `search`) samples /proc/loadavg at run start and refuses when
  the 1-min load average exceeds `--max-load` (default: half of available
  CPUs — catches the F-006 class, load 24 on 16 cores, while tolerating
  desktop noise). Every output record carries `load`/`load5`/`load15`/
  `max_load`/`load_override`; `--allow-loaded` runs anyway and sets
  `load_override: true`, keeping overrides auditable.
- regression: `load_guard_tests` (refusal/override/NaN decision table,
  default scaling, CLI flags on all four benchmark subcommands);
  live refusal (exit 1) and override demonstration logged in
  trials/f006-load-guard.jsonl (git 282577c).
- historical note: pre-guard records (incl. the a8bf146 exam) keep their
  batch-scoped load; only new runs carry per-run load (closes F-021's
  remaining structural gap).

## F-007 selector behind same-equation gristmill on confirmed ADC2/EOM cases [OPEN — ADC2 mechanism verified]
- symptom: the valid a8bf146 exam measures six-problem geomean 0.902x t1 /
  0.883x t16 vs same-equation gristmill. Complete speedups: MP3 1.105/1.041;
  CCD 0.996/0.936; CCSD 0.947/1.201; CCSD-lambda 1.042/0.580;
  EOM 0.794/0.811; ADC2 0.626/0.862 (t1/t16). The CCSD and lambda thread
  inversions have NO verified mechanism (F-019). Axis 1 fails on all six
  (FLOP ratios 1.258x to 1397.562x; F-018). The older sub-protocol archive
  remains retracted; these numbers replace it. Earlier "1.42x CCSD win"
  was a baseline artifact — also retracted.
- mechanism: VERIFIED on the ADC2 singles residual (11 terms). The selected
  schedule reproduces exactly and stops after one parenthesization; its
  unfactored three-factor contractions dominate the per-op profile. Gristmill
  instead retains alternative parenthesizations while optimizing the sum
  (its default `ContrStrat.TRAV`) and reaches a better combination of splits
  and shared intermediates. This is a greedy-path failure, not a missing
  parenthesization action or a final-objective inversion: at (16,64), t1,
  the model ranks gristmill better (105.1 vs 119.9 ms predicted; 58.7 vs
  112.7 ms measured), but after the first accepted move no remaining split
  clears the single-move acceptance rule. FLOP-greedy also stops at a worse
  local optimum. EOM-CCSD has the same final-schedule signature (920.3 vs
  1162.1 ms measured; model correctly ranks gristmill better), but its
  selector reproduction timed out after 240 s / three steps, so an identical
  path mechanism there is not yet claimed.
- v2-era state (exam-cd921bd, debug binary pinned): under model v2 + the
  F-018 cap, five of six regenerated candidates are baseline fallbacks, so
  their Axis-2 cells are same-program noise (+-9%: ccd/ccsd/lambda/eom/adc2
  ratios 0.949-1.089 around 1.0). The old losses (adc2 0.626x, eom 0.794x,
  lambda t16 0.580x) are gone BY CONSTRUCTION — the compiler no longer ships
  them — but the same fact means the v2-priced greedy selector finds nothing
  better than baseline on ccd/ccsd/adc2/eom/lambda. mp3 is the exception:
  2 accepted steps, FLOPs 0.969x BELOW baseline, measured 1.194x t1 /
  1.139x t16. Open question reframed: is v2-greedy search quality (not the
  v1-era acceptance hole) the binding constraint on Axis 2?
- exposing fixture: `adc2_selected.json` vs same-equation
  `adc2_gristmill.json`; exact selector reproduction is
  `harness select benches/chemistry/adc2.json --sizes 0=16,1=64 --threads 1`.
- data: archive/trials-v1/final_vs_gristmill.jsonl;
  trials/f007-per-op-diff.jsonl
- NOTE: this entry is also the workflow's first validation — the
  same-equation-baseline discipline caught a false win before publication.
- fix candidates: no active candidate. I-005 was retired on valid train
  controls; I-001 remains untested because the diff did not measure the
  candidate conflicts required by its wins-IF condition.

## F-008 unconditional definition chunking destroys cross-chunk factorization [CLOSED — split threshold aligned with kernel envelope]
- symptom: the current default one-step selector no longer reproduces the
  archived CCD candidate and leaves near-reference FLOP counts. No CCSD
  chunking claim is made: its unsplit default-guard control was invalid
  (F-010), and its archived command provenance is incomplete.
- mechanism: VERIFIED by controlled reproduction. Commit 2b745e7 made the
  selector split every definition above 12 terms to bound biclique search.
  The chunks are optimized independently, so factorizations spanning chunks
  disappear. Running the current selector with splitting disabled restores
  the archived terminal quality (0.7822 vs 1.0929 s predicted; log10 FLOPs
  10.019 vs 12.644 under chunking). The original 967e26a binary independently
  reproduces the archived candidate hash. A separate-full/variant-prefilter
  experiment did not improve terminal quality and was reverted.
- exposing fixture: `harness select benches/chemistry/ccd.json --sizes
  0=16,1=64 --threads 1 --split-above 0`; compare with default chunking and
  `benches/candidates/ccd_selected.json` from commit 967e26a.
- fix: the default split threshold is now 20, matching the default kernel
  skip envelope. CCD's 18-term definition is no longer chunked. Larger defs
  remain an F-005 problem; this is not a general enumeration-budget fix.
- regression: default CCD selection reproduces the unsplit nine-step trace
  and 0.7822403455 s terminal prediction; final emission uses the F-018
  baseline fallback because that schedule still exceeds Axis 1. Permanent
  default-threshold unit test + numerical oracle on the emitted baseline.
  NOTE (model v2, 0b44e61): the nine-step trace was produced under v1
  pricing; under v2 rates the CCD selector converges at step 0 and emits
  the baseline fallback directly (trials/model-v2.jsonl). The structural
  fix (split threshold aligned with kernel envelope) stands; the trace
  claim is historical to v1 pricing.

## F-009 MP3 selected schedule fails the numerical oracle [CLOSED]
- symptom: protected milestone Axis 0 stopped on `mp3_selected`: exact
  `truthmill::verify` passed, but the numerical oracle failed with worst
  relative difference 1.984844851549236 at sizes (3,5), model layout.
- mechanism: VERIFIED. Canonicalization uses composed symmetry-group
  elements, but input generation propagated only along edges whose endpoint
  was already a used access pattern. Two used blocks related through an
  unused intermediate were independently random. Direct bridge roundtrip was
  correct; State canonicalization changed single-term numerical values while
  exact verify accepted them. A full-group regression failed on composed
  permutation [1,0,3,2] before the fix.
- fix: generate the complete access-pattern orbit, including unused
  intermediates, so every composed cross-block relation is enforced.
- exposing fixture / regression: `dense::tests::generated_mp3_inputs_
  satisfy_declared_symmetries`, direct and canonicalized MP3 bridge-oracle
  tests, and the original `harness check ... mp3_selected.json` command.
- result: MP3 selected passes layout-none/model oracle; full cargo test
  passes. Full-size scalar oracle was not used (exponential; F-015 convention).

---

## F-026 exam-cd921bd driver omitted ccsd_t from Axis 0 [CLOSED — addendum recorded, script fixed]
- symptom: tools/run_exam.py phase B skipped ccsd_t pairs entirely, while
  the a8bf146 exam covered all 14 pairs; only ccsd_t TIMING is
  memory-blocked (F-017) — its (3,5) Axis-0 checks are cheap and required.
- mechanism: my exclusion logic conflated the timing blocker with the
  correctness gate. Found by self-review before the milestone closed.
- fix: phase B includes all problems; the two omitted checks ran as a
  recorded addendum (both pass, worst rel 2.5e-15) in
  trials/exam-cd921bd.jsonl (kind=protected-exam-axis0-addendum).
- regression: run_exam.py phase B covers all seven problems.

## F-035 Backend n-ary execution planning is nondeterministic per process; measured wall times can differ for identical programs [OPEN]
- symptom: F-028's mechanism applies to EXECUTION too: interp::prepare
  (ConcreteEinsumPlan::prepare_subscripts) and eager einsum plan through
  ContractionTree::optimize -> omeco TreeSA, whose tie-breaking is
  per-process random. Two `harness time` invocations of the SAME candidate
  can execute different contraction trees for n-ary ops, moving measured
  wall time without any code or input change. Candidate contributor to the
  same-program exam noise (F-027: identical-program cells diverge up to
  8.9%) and the F-022 anchor drift — neither is claimed explained, just
  exposed to a mechanism neither considered.
- verification status: planning nondeterminism itself is VERIFIED (F-028
  probes: different trees per process for the same op). Its wall-time
  impact on corpus problems is UNMEASURED — needs per-process repeated
  timing of an n-ary-heavy candidate (e.g. eom_ccsd) across several
  processes at fixed load.
- constraint: tenferro 0.7.1's public concrete-plan API has no
  explicit-path constructor (EinsumOptimize::Tree/Path exists only on the
  traced path), so pinning execution trees needs either an upstream change
  (I-004 material: omeco IncidenceList ordering + rng seeding) or
  harness-side lowering of n-ary ops to pinned binary chains (interacts
  with F-002's lesson and I-002's tree slot — decide at M2, not ad hoc).
- model fidelity note: post-F-028 the model prices det_greedy_pairs trees;
  execution follows TreeSA's per-process trees for n-ary ops. The
  difference is bounded by tie width (equal-complexity orderings), now a
  stated assumption rather than an implicit one.
- evidence addendum (F-033 batch): release-binary mp3 gristmill t16 is
  bimodal across processes (50.8 vs 58.0-58.2 ms, 3 reps) — consistent
  with per-process tree rolls moving real wall time (selected is stable
  at ~52 ms). First direct timing signature of this defect.

## F-036 Migrate calibration and exams to the release binary profile [OPEN]
- symptom (F-033): all calibration rates (calibrate-v1) and both protected
  exams were measured with the debug-profile harness; release A/B shows
  debug does not preserve ratios for the one genuine non-fallback cell
  (mp3: 1.19x debug t1 -> 1.02-1.04x release t1; baseline gains more from
  release than the selected schedule). SPEC measures the compiler's
  artifact; a debug harness is not it.
- plan: (1) release-profile `harness calibrate` -> calibrate-v2 rates file;
  (2) cost model rates refresh (a model change -> fires a milestone); (3)
  re-run the protected exam on release. Sequenced AFTER I-007 (I-007 is
  model-priced and internally consistent under debug-calibrated rates).
- constraint: debug-era absolute ms remain valid within the debug regime;
  cross-profile comparisons of absolute times are forbidden.

## Critic challenges (milestone after I-005 retirement; filed by fresh-context critic, milestone stays OPEN until each is fixed or rebutted with evidence)

## F-010 I-005 retirement rests on an invalid CCSD control [CLOSED — corrected and rerun]
- symptom: I-005 was initially retired on an unsplit CCSD control that
  skipped its 63-term definition under the default >20-term guard; the ADC2
  wins-IF fixture had not run and the CCD kill margin was model-only / 0.8%.
- mechanism: VERIFIED by the critic. The 0.11 s CCSD compile demonstrated
  that the dominant definition was skipped, so zero accepted moves was not
  evidence about lookahead.
- data: trials/i005-lookahead.jsonl; crates/harness/src/main.rs:218-228.
- resolution: accepted. Reran CCSD with `H_BICLIQUE_MAX_TERMS=1000`, ran the
  ADC2 wins-IF fixture, and measured CCD/CCSD with 10 iterations / 3 warmup.
  I-005 is retired on the resulting complete evidence (IDEAS.md result and
  superseding trial in trials/i005-lookahead.jsonl), not the invalid controls.

## F-011 F-008 "splitting disabled restores archived quality" verified only for CCD; CCSD counter-evidence unremarked [CLOSED — scope corrected and evidence logged]
- symptom: F-008's mechanism states disabling chunking "restores the
  archived terminal quality" and the symptom names CCD/CCSD, but every
  number cited (0.7822 vs 1.0929 s; log10 FLOPs 10.019 vs 12.644) is CCD.
  The only recorded CCSD unsplit number is 2.6623 s predicted — WORSE than
  the chunked control 2.2169 s (trials/i005-lookahead.jsonl) — the opposite
  direction, unmentioned in F-008 or RESEARCH.md. (Likely a skipped-defs
  artifact per F-010, but then the CCSD half of the claim is simply
  unverified either way.)
- unlogged supporting claims: the log10-FLOPs pair (10.019/12.644) appears
  in no trial line; "the original 967e26a binary independently reproduces
  the archived candidate hash" records no hashes and no trial line; "a
  separate-full/variant-prefilter experiment did not improve terminal
  quality and was reverted" has no trial line and no numbers.
- resolution: accepted. F-008 is now CCD-only and the CCSD counter-evidence
  is explicit. The superseding trial records FLOPs, both SHA-256 hashes, and
  the prefilter outcomes; the reverted prefilter makes no mechanism claim.

## F-012 F-007 comparative numbers rest on sub-protocol measurements [CLOSED — unsupported aggregate retracted]
- symptom: "geomean 0.90x t1 across six problems; losses 0.77x/0.67x; wins
  mp3, ccsd t16" cites archive/trials-v1/final_vs_gristmill.jsonl, where
  every line is iters=4, warmup=2 (protocol: median >=10 after >=3 warmup)
  and carries no load field (post-F-006 schema requires load). Batch-level
  noise is visible in the file itself: adc2_selected t16 medians are
  273.7 / 80.3 / 178.1 ms across repeats of the same candidate (3.4x
  spread); adc2_gristmill t16 spans 77.8–150.3 ms.
- status of corroboration: only adc2 and eom_ccsd t1 were re-measured
  under load recording in trials/f007-per-op-diff.jsonl. The ccd, ccsd,
  ccsd_lambda, and mp3 components of the geomean — and the ccsd t16 "win" —
  have no guard-compliant measurement. Given F-006 (a contaminated batch
  once invalidated an entire era), the 0.90x figure is not currently
  supported to protocol standard.
- resolution: accepted. The archived aggregate was retracted. A later valid
  a8bf146 exam independently supplies guard-compliant six-problem numbers;
  CCSD(T) is excluded because its memory guard stopped that exam.

## F-013 mp3 "win" was timed on a candidate that fails the numerical oracle [CLOSED — win retracted]
- symptom: F-007's symptom and RESEARCH.md count mp3 as a selector win;
  F-009 shows mp3_selected fails the numerical oracle (worst rel diff
  1.98). A win/loss/geomean computed on a schedule whose correctness is
  open is not evidence.
- the signal predates the milestone: in final_vs_gristmill.jsonl,
  mp3_selected's checksum (28.799) differs from mp3_gristmill's (31.283) in
  every line, while the ccd, ccsd, ccsd_lambda, eom_ccsd, and adc2 pairs
  all match exactly. The divergence was visible in the archived data and
  was never connected to the timing claims until the exam stopped on it.
- resolution: accepted. The archived MP3 win was retracted. After F-009 was
  fixed, the a8bf146 exam passed Axis 0 and independently measured valid MP3
  speedups of 1.105x t1 / 1.041x t16.

## F-014 Provenance violation: I-005 test ran on uncommitted code; run configuration unrecorded [CLOSED — source committed and record superseded]
- symptom: trials/i005-lookahead.jsonl records git 6ee4eee, but the
  `--lookahead` implementation (~243 lines in crates/harness/src/main.rs)
  exists only in the dirty working tree; commit 6ee4eee adds 11 lines and
  no lookahead code. The ledger edits recording the retirement (FAILURES,
  IDEAS, RESEARCH) are likewise uncommitted at critic time. The decisive
  experiment for a retired idea is not reproducible from the repository.
- also unrecorded: the value of `H_BICLIQUE_MAX_TERMS` (decisive for
  F-010), the exact select invocations, and `ccd_unsplit_one_compile_s` —
  so the adoption rule's compile-time gate (<10x one-step) is unevaluable
  on the only valid (CCD) control, and the recorded CCSD ratio
  (2.94/0.11 = 27x) is an artifact of skipped defs.
- protocol reference: trials/README requires every line to carry the
  commit hash at run time; PROTOCOL gate 5a (cargo test) has no record for
  the selector change.
- resolution: accepted. The exact experimental source is commit 4d04b21;
  CCD control and ADC2 were reproduced after commit. A superseding trial
  records explicit commands, `H_BICLIQUE_MAX_TERMS=1000`, all compile times,
  load, oracle results, and cargo-test success. The original dirty-source
  line remains as audit history, not as the basis for adoption.

## F-015 Protected exam ran at non-spec sizes and produced no axis information beyond one gate failure [CLOSED — Axis-0 size convention documented]
- symptom: the milestone exam (trials/i005-lookahead.jsonl) ran Axis 0 at
  sizes (3,5). SPEC defines corpus sizes {(8,40),(16,64),(32,96)} and no
  Axis-0 size convention; no document records why (3,5) is the exam size.
  The exam stopped at the first failure, so it yielded no Axis-1/2 numbers
  and no pass/fail for the other six candidates; there is also no
  tournament record (even a vacuous "no contested slot evaluable" line).
  As recorded, the milestone's protected evaluation does not correspond to
  the spec'd evaluation at any documented size.
- resolution: accepted. SPEC now documents why Axis 0 uses (3,5): the
  independent scalar oracle is exponential and full-size brute force is
  infeasible (a (16,64) MP3 diagnostic hit the 300 s limit). Stopping at the
  first failure is correct gate behavior; no Axis-1/2 claims were made. A
  tournament record exists in the superseding trial. The F-009 fix is a new
  Axis-0 event and permits one fresh protected exam.

## Critic challenges (milestone after the a8bf146 protected exam; filed by fresh-context critic, milestone stays OPEN until each is fixed or rebutted with evidence)

## F-018 Axis-1 gate fails on all six measured problems — including the entire train corpus — and no F-entry owns the gate failure [CLOSED — emission cap and baseline fallback]
- symptom: trials/exam-a8bf146.jsonl records axis1_pass=false, failures=6,
  FLOP ratios from 1.2581x (ccd) to 1397.5622x (ccsd_lambda). The train
  corpus fails too: mp3 1.6247x, ccd 1.2581x, ccsd 3.9981x. SPEC Axis 1:
  "<= 1.05x the same-equation gristmill baseline's count, every corpus
  problem. This is a REAL gate ... it holds for the incumbent engine by
  construction." The exam falsifies "by construction": the incumbent's own
  selected schedules fail the gate on every measured problem. This
  contradiction between SPEC text and measured reality is filed nowhere.
- mechanism ownership gap: F-007's VERIFIED mechanism covers the ADC2
  singles residual (11 terms) only; the EOM entry explicitly declines to
  claim a mechanism. The ccsd_lambda (1397x), eom_ccsd (39x), ccsd (4.0x),
  mp3 (1.62x), and ccd (1.26x) FLOP gaps have no verified mechanism and no
  dedicated F-entry; a corpus-wide gate failure is carried as a symptom
  inside an entry scoped to "confirmed ADC2/EOM cases."
- unrecorded configuration skew: the I-005 supersession records ccsd
  one-step FLOP ratio 5.0185x (unsplit, H_BICLIQUE_MAX_TERMS=1000) while the
  exam's ccsd row shows 3.9981x — the exam's timed candidates were generated
  under selector configurations that the exam line does not record, so the
  diagnosis numbers and the exam numbers are not known to describe the same
  schedules.
- response: accepted. This entry owns the corpus-wide Axis-1 failure; SPEC no
  longer claims incumbent green by construction. Only ADC2 has a verified
  mechanism; MP3/CCD/CCSD/lambda/EOM mechanisms remain open work. Historical
  selector configurations are unknown, so I-005 and exam FLOP ratios are not
  asserted to describe identical schedules. Structured run records now pin
  every timed candidate by SHA-256.
- mechanism / fix: VERIFIED that selector acceptance priced only wall time and
  never checked Axis 1. MP3 demonstrates the tradeoff: 1.625x FLOPs but 1.105x
  measured speedup. `harness select` now loads a verified same-equation
  baseline, enforces <=1.05x FLOPs at final emission for kernel AND structural
  engines, and falls back to that baseline when search misses the cap. This is
  a feasibility fix, not a search-quality claim; F-007/F-019 remain open.
- regression: `axis1_tests::over_cap_mp3_schedule_falls_back_to_baseline`;
  zero-step searches on all seven problems emit JSON-equivalent baselines and
  pass the numerical oracle.

## F-019 F-007's symptom selectively characterizes the exam: unremarked losses and unexplained t1/t16 inversions [CLOSED — complete results reported, inversions owned]
- symptom: F-007 names losses "ADC2 0.626x t1, EOM 0.794x t1, CCSD-lambda
  0.580x t16" and wins "MP3 and CCSD t16". The exam rows show 7 of 12 cells
  below 1.0 — additionally ccd t16 0.9363x, ccsd t1 0.9471x, adc2 t16
  0.8623x, eom_ccsd t16 0.8108x, none remarked. Two thread-regime inversions
  have no recorded mechanism or even hypothesis: ccsd loses t1 (0.9471x) but
  wins t16 (1.2006x); ccsd_lambda wins t1 (1.0421x) but loses t16 (0.5799x).
  RESEARCH.md inherits the selective subset ("MP3 and CCSD-t16 wins are
  supported; ADC2/EOM losses ... are supported"), rendering ccd/ccsd mixed
  results invisible in the synthesis.
- resolution: accepted. F-007 and RESEARCH.md now report every t1/t16 cell.
  CCSD and CCSD-lambda thread inversions are explicitly unmeasured mechanisms,
  not implied wins or layout effects.

## F-020 I-005 retirement: the pre-registered adoption rule was evaluated on incomplete problem sets [CLOSED — incomplete clauses marked unevaluated, Axis-1 clause dispositive]
- symptom: the adopt rule reads "train geomean measured wall time improves
  >5%, no train problem regresses >2%". Train is {mp3, ccd, ccsd}. The
  superseding trial (trials/i005-lookahead.jsonl, 4d04b21) measures only ccd
  and ccsd; mp3 was never timed under either arm and no challenger FLOP
  ratio is recorded for it. The recorded train_geomean_speedup 1.0367 is a
  two-problem geomean labeled as the train geomean; for the >5% clause to
  have passed, the unmeasured mp3 arm would have needed a >7.7% lookahead
  win, so that clause is unevaluated, not failed. The idea text also says
  "evaluate ADC2/EOM once at the milestone"; EOM was never evaluated for
  I-005. The disposition "retired_mechanically" is rescued only by the
  Axis-1 clause (challenger ratios ccd 1.2648x, ccsd 1.5996x, both red),
  which is sufficient — but the recorded evidence does not cover the rule
  as written, and no trial line states that mp3/EOM were deliberately
  skipped or why.
- resolution: accepted. The 1.0367 value is now labeled a two-problem
  geomean; MP3/EOM and the full train-geomean clause are unevaluated. I-005
  is retired solely because its measured CCD/CCSD candidates independently
  violate the Axis-1 adoption clause. The tournament record states this.

## F-021 The a8bf146 exam record does not institutionalize the F-006/F-013 lessons: sizes, iteration counts, checksums, and per-benchmark load are all absent [CLOSED — per-run load now structural]
- symptom: the protected-exam-axis1-axis2 line records no problem sizes
  (SPEC's corpus has three; which one produced the rows is knowable only by
  inference from F-007 fixtures), no iters/warmup fields (the protocol's
  ">=10 after >=3 warmup" compliance exists only as note prose, the same
  attest-in-prose pattern whose absence made F-012's retraction necessary),
  and no per-run output checksums — F-013's false win was detectable
  precisely in archived checksum divergence, and the new exam schema drops
  that signal. One load triplet covers a six-problem x two-candidate x
  two-thread batch and is byte-identical to the Axis-0 line's load,
  i.e. sampled once and copied rather than recorded "at start" of each
  benchmark as trials/README specifies. The f007-per-op-diff line uses yet
  another load schema (single scalars per phase). The structural fix F-006
  names (runner refuses under load, records load per run) remains unbuilt.
- response: accepted. trials/exam-a8bf146-runs.jsonl adds sizes, 10/3
  iteration/warmup fields, min/median/max, checksums, and candidate hashes
  for all 24 runs. Batch-start load is explicitly scoped as such; the
  a8bf146 records were not retrofitted.
- resolution: the F-006 structural guard (git 282577c) makes every timing
  command sample and record load at run start and refuse above
  `--max-load`, so all future exam records carry per-run load by
  construction. Historical records are unchanged, not fabricated.

## F-022 The ADC2 anchor baseline differs ~15% across two guard-compliant sessions; unremarked, and F-016's calibration depends on which is right [CLOSED — variance recorded, calibration claim scoped]
- symptom: adc2 same-equation gristmill t1 measured 58.7 ms in
  trials/f007-per-op-diff.jsonl (f16cb09, load 1.32) vs 67.929 ms in the
  a8bf146 exam (load [0.55,1.22,1.33]) — 15.6% apart on the anchor baseline
  of the headline loss. adc2_selected agrees within 3.9% (112.7 vs 108.5),
  so the discrepancy is not uniform session drift. The headline ADC2 loss
  magnitude (0.6261x) and F-016's recorded model error ("predicted gap
  1.14x vs measured 1.92x") both use the 58.7 ms figure; F-016 is now an
  M1 calibration exposing case, yet which session anchors it has never
  been examined. No entry records the discrepancy.
- resolution: accepted. Both sessions are now explicit: gristmill 58.7 vs
  67.929 ms and selected 112.7 vs 108.492 ms. F-016 uses a measured-gap RANGE
  of 1.60–1.92x vs 1.14x predicted, not a privileged baseline session. The
  underprediction survives either anchor; no cause for the variance is claimed.

## F-023 The protected exam ran with a known-blocking memory cap, spending the run-once budget on a void ccsd_t component [CLOSED — timeline rebutted, sequencing recorded]
- symptom: F-017 was OPEN with a VERIFIED mechanism before the exam fired:
  the harness default 6000 MB cap sits below ccsd_t's (8,32) estimated peak
  8587 MB, which itself sits below SPEC's 14 GB envelope. The exam
  predictably stopped at the ccsd_t gristmill t1 baseline guard, produced
  no ccsd_t timing, and is reported incomplete. PROTOCOL grants one exam
  run per event and forbids iterating it; that single run was consumed on
  a configuration whose ccsd_t failure was known in advance. No ledger
  entry records that this sequencing was a deliberate trade-off (versus
  deferring the exam until the planned F-017 cap fix, or setting the
  SPEC-consistent cap beforehand), so the decision is unauditable.
- resolution: the premise is rebutted by the repository timeline. F-017 was
  filed AFTER this exam first observed the baseline's 8587 MB estimate; no
  prior baseline measurement established that exact blocker. The run was not
  a deliberate trade-off and was not rerun. F-017 owns the cap for the next
  milestone.

## F-024 Provenance recurrence: exam log untracked and responding ledger edits uncommitted at critic time [CLOSED — exam evidence and responses committed]
- symptom: F-014 (CLOSED) established the class — decisive records not
  reproducible from the repository — and its resolution stated "future
  trials require committed source." At this critic's reading: the exam ran
  on committed source (a8bf146, satisfying the narrow resolution), but
  trials/exam-a8bf146.jsonl is untracked and the FAILURES.md/RESEARCH.md
  updates responding to the exam are uncommitted working-tree
  modifications. The audit-trail half of the F-014 lesson recurred within
  one milestone. Related ledger-integrity nit: F-011's entry ends with the
  sentence "A mechanism marked VERIFIED rests partly on unlogged
  experiments" appended AFTER its resolution paragraph — a fragment of the
  original challenge left dangling so the closed entry contradicts itself.
- resolution: accepted. The exam logs, structured run records, tournament
  disposition, and these ledger responses are committed together in the
  milestone-response commit. F-011's dangling challenge fragment is removed.

## F-025 Tournament record covers only I-005; standing slot challengers have no disposition, and the displacement rule is currently unsatisfiable [CLOSED — complete disposition recorded]
- symptom: IDEAS.md's contested factorization slot names I-001 and "the
  structural proposer" as standing challengers. The milestone tournament
  line (trials/i005-lookahead.jsonl, 4d04b21) records only I-005 — which
  was a selector lookahead, not a listed slot challenger. I-001 (proposed,
  never priced) and the structural proposer (named a challenger but never
  given an I-entry, despite ccd_struct/ccsd_t_struct/etc. candidates
  existing in benches/candidates/) received no tournament disposition, not
  even a "not evaluable" line. Separately, the slot's displacement rule
  requires a challenger to "keep Axis 1 green (<=1.05x gristmill FLOPs on
  every corpus problem)" — a condition the incumbent itself fails on all
  six measured problems (exam-a8bf146). The contested-slot rules as
  written presume a green incumbent; no ledger entry notes that the rule
  is presently unsatisfiable or what it means while Axis 1 is red.
- resolution: accepted. trials/tournament-a8bf146.jsonl records I-001 and
  the structural proposer as NOT EVALUABLE and I-005 retired. IDEAS.md states
  that the incumbent is Axis-1 red and no displacement tournament occurs
  until that gate is restored; no challenger is silently adopted or retired.

## F-016 Held-out model error is untracked; the model that killed I-005 mispredicts ADC2's gap by ~70% [OPEN — response recorded, calibration owned by M1]
- symptom: trials/f007-per-op-diff.jsonl shows ADC2 predicted gap
  119.9/105.1 = 1.14x vs measured 112.7/58.7 = 1.92x. F-004 covers only
  the ccsd_t high-rank slow path; no F-entry records that the current
  model badly under-prices the selected-vs-gristmill gap on adc2-class
  schedules. RESEARCH.md cites ranking rho ~0.93 measured on ccd/ccsd
  (train) only. The same model was used to kill I-005 at a 0.8% margin
  (F-010) and to rank all F-007 mechanisms; its held-out quantitative
  error has no calibration record and no owner.
- response: accepted as an M1 calibration requirement. ADC2's 1.14x predicted
  vs 1.60–1.92x measured gap RANGE (two load-recorded sessions, F-022) is an
  explicit F-004/F-016 exposing case. I-005 was not killed on that model
  margin: measured CCD/CCSD Axis-1 failure forced retirement.
  No held-out calibration or parameter tuning is permitted; evaluate model v2
  on this case only at a milestone.
- model v2 status (0b44e61): the F-004 reduction-class hole is closed and
  CI-gated (model-ci, 2x, exceptions documented). Newly surfaced PRE-EXISTING
  gap classes, measured on the train corpus at (8,40) t1 (diagnostic-grade,
  trials/model-v2.jsonl): (a) k-heavy low-rank-output contractions
  (ccsd_gristmill op#19 "ab,bacd->cd") achieve 0.26 GF/s — 500x under-
  predicted; (b) single-k rank-4-output contractions (op#23 "ab,cdae->bcde")
  achieve 0.9 GF/s — 26x under-predicted; (c) rank-6 lincomb ops carry
  ~0.3-0.5 ms fixed overhead the dispatch floor misses (10-17x at tiny size,
  ~5% of ccsd_t_struct time). Per-op spearman on ccsd_gristmill is 0.37
  (n=115, dispatch-floor-dominated); the prequel's rho ~0.93 was measured on
  different fixtures — not asserted comparable. Next M1 work: extend the
  calibration grid with classes (a)-(c), then price them.

## F-017 CCSD(T) exam blocked by harness 6 GB limit below the 14 GB regime [OPEN]
- symptom: the a8bf146 protected exam reached ccsd_t gristmill t1 and stopped
  before timing: estimated peak 8587 MB (inputs 137 MB) vs default 6000 MB.
- mechanism: VERIFIED by the runner guard. SPEC allows this machine's 14 GB
  envelope and selects ccsd_t at (8,32) to fit it, but `harness time` defaults
  to a smaller 6000 MB cap. The baseline was not run and no timing exists.
- exposing fixture: `harness time benches/chemistry/ccsd_t.json --sizes
  0=8,1=32 --threads 1 --candidate benches/candidates/ccsd_t_gristmill.json
  --layout model` with default `--max-mem-mb`.
- resource blocker (2026-10-30): conservative probes beyond the 6 GB guard
  attempted additional 4 GiB backend allocations and aborted at ~9 GB RSS;
  swap filled (~4 GB), and a later factmem probe was terminated. Raising the
  address-space cap did not make the baseline viable. All limit-raising and
  CCSD(T) execution experiments were reverted; do NOT rerun full-size CCSD(T)
  on this machine. The a8bf146 exam remains incomplete as-is.
- planned fix: memory-safe schedule/transform development only on tiny
  fixtures with a hard allocation budget, followed by static estimation.
  No further full CCSD(T) execution until a schedule is statically shown to
  fit well below the existing 6 GB guard.

## Critic challenges (milestone after exam-cd921bd; filed by fresh-context critic, milestone stays OPEN until each is fixed or rebutted with evidence)

## F-027 Exam Axis-2 batch ran under climbing load; identical-program cells diverge up to 8.9%; the milestone's only genuine win is barely above demonstrated same-batch noise and is unreplicated [CLOSED — replicated on debug; does NOT survive release, see F-033/F-036]
- resolution: replicated. trials/f027-f033-replication.jsonl: mp3 selected
  vs gristmill on the exam's debug binary, 3 interleaved A/B rounds at
  load ~1.1-1.7: t1 speedups 1.1857/1.1855/1.1800 (exam: 1.1937), t16
  1.1838/1.0972/1.1200 (exam: 1.1393). The win replicates on the exam's
  own binary and clearly exceeds the same-batch identical-program noise
  (<=1.09x) at t1. HOWEVER the F-033 release check shows the win is
  profile-dependent (release: 1.017-1.045x t1, t16 roll-dependent), so
  RESEARCH.md now calls it a debug-regime win only. The load-climb and
  within-run dispersion observations stand as recorded; the load guard's
  start-of-run sampling blind spot is acknowledged (mid-run contamination
  remains visible only in min/max fields).
- symptom: exam-cd921bd load_at_start [0.75,1.24,1.49] vs load_at_end
  [7.69,4.37,3.25]; per-run start loads range 1.74-5.58. Within single
  10-iter runs the recorded min/max fields show adc2 t16 spans 93.96-344.06
  ms (gristmill) and 94.20-198.36 (selected) — 3.7x dispersion; ccsd_lambda
  t16 spans ~1.6x. The five fallback-vs-baseline cells are the SAME
  computation timed twice (identical log10_flops and checksums), yet their
  medians diverge up to 1.089x (adc2 t16 144.10 vs 156.88; ccsd t16 654.51
  vs 710.72 = 1.086x). That is a direct same-batch measurement of
  noise-on-identical-programs — and the exam's only genuine Axis-2 win at
  t16 (mp3 1.139x) barely exceeds it; only mp3 t1 (1.194x) clearly does.
  Neither mp3 cell was replicated in a second session; F-022 documented
  15.6% cross-session drift on a baseline anchor. RESEARCH.md reports the
  fallback cells as "noise (+-9%)" without connecting that measurement to
  the significance of the mp3 win, and no entry records the adc2 within-run
  3.7x dispersion.
- guard gap: the F-006 load guard samples only at run start; mid-run
  contamination is invisible to it by design and is visible here in the
  recorded min/max fields.
- data: trials/exam-cd921bd.jsonl, trials/exam-cd921bd-runs.jsonl.

## F-028 Selector is nondeterministic across identical-source invocations; unremarked [CLOSED — model-side deterministic planner; execution side filed as F-035]
- symptom: phase-A candidate regeneration ran twice (voided a80b14e
  attempt, then cd921bd). Harness source is byte-identical between those
  commits (git diff a80b14e..cd921bd touches only tools/run_exam.py), yet
  the ccsd_lambda traces diverge: step 0 is "paren def0 term19 cut1" vs
  "cut2", and step-1 terminal predictions differ (1.3467922 vs 1.3470174
  s). The final candidate sha256 matched (9819962d4525e496) only by tie
  convergence; equal-scoring candidates appear to be enumerated/broken
  nondeterministically. Hash-pinned provenance (F-018/F-021/F-024
  responses) and the fallback-equivalence regression tests presume a
  deterministic emitter; that the search itself is nondeterministic is
  recorded nowhere.
- mechanism: VERIFIED end-to-end. (1) Reproduced on mp3 with the committed
  probe (`harness select-probe`): the SAME initial-state computation hash
  scored 0.397 s vs 0.621 s across processes, and repeated scoring within
  one process was non-idempotent (rep0=rep1!=rep2). (2) The divergence is
  inside the model's tree walk: tenferro's ContractionTree::optimize routes
  >=3-operand einsums to omeco TreeSA; its greedy initializer iterates
  HashMap-backed incidence lists (omeco-0.2.6 greedy.rs), so
  contraction-cost ties resolve in per-process/per-call random order (std
  RandomState advances per HashMap::new(), explaining the in-process
  non-idempotence). Observed consequence: the same n-ary op got different
  contraction trees (swapped step order, swapped lhs/rhs) with different
  clean/reduce classifications.
- fix: the model no longer calls ContractionTree::optimize. costmodel.rs
  pins the tree with `det_greedy_pairs`, a faithful port of tenferro's
  private deterministic `optimize_self_greedy_pairs` (set-membership
  counts, first-appearance intermediate order), then
  ContractionTree::from_pairs. Selector/model scoring is now deterministic
  (mp3 select traces and full records bit-identical across processes;
  model-ci still passes).
- regression: `determinism_tests::model_scoring_is_deterministic_across_
  repeated_calls` (fails pre-fix: in-process scores differed); H_TRACE
  stream diffing via the committed select-probe instrument.
- SCOPE NOTE: this pins the MODEL. Execution planning (interp::prepare,
  eager einsum) still routes through the same nondeterministic backend
  path per process — that defect and its timing consequences are F-035.
  The model now prices a deterministic canonical tree that execution may
  not follow for n-ary ops; the fidelity gap is recorded in F-035.
- data: trials/exam-a80b14e-runs.jsonl vs trials/exam-cd921bd-runs.jsonl
  (ccsd_lambda phase-A lines); repro streams referenced in the f028 trial
  line.

## F-029 "Axis 1 green" is a 6/7 claim; ccsd_t was timed on a stale v1-era candidate with no hash pinning, and its slowness mechanism is asserted without a profile [CLOSED — ccsd_t RED recorded; attribution profiled]
- symptom: exam phase C flop_ratios lists six problems; ccsd_t — a corpus
  problem whose Axis-1 check is execution-free — is absent. F-026 closed
  the identical omission for phase B only; the phase-C half persists and is
  recorded nowhere in the exam line, the milestone commit message, or
  RESEARCH.md ("Axis 1 is green for the first time", "all six <=1.05x").
  Phase D timed benches/candidates/ccsd_t_selected.json, last committed at
  967e26a (2026-10-28, v1 pricing era, pre-F-018 cap, never regenerated
  under v2; sha256 250d550e... matches the a8bf146 pin). Neither ccsd_t
  timed candidate carries a candidate_sha256 anywhere in the cd921bd
  records, so the F-021/F-018 resolution ("every timed candidate pinned by
  SHA-256") does not hold for the problem where the compiler is most likely
  losing. RESEARCH.md attributes the 27.1 s/iter t1 to "the
  reduction-class slowness in vivo"; the reduction-class mechanism was
  verified on ccsd_t_struct at (4,8) and on the calibration grid (F-004),
  but no per-op profile of THIS candidate at (8,32) exists in trials/ —
  the attribution for the shipped candidate is asserted, not verified.
- data: trials/exam-cd921bd.jsonl phase C/D ccsd_t rows; git log
  benches/candidates/ccsd_t_selected.json.
- resolution (partial, trials/f029-ccsd_t-axis1.jsonl): accepted. ccsd_t's
  Axis-1 check is execution-free and now recorded: the stale v1-era
  ccsd_t_selected (sha 250d550e22625247, matching the critic's pin) is
  Axis-1 RED — 40.6x cap at (8,32), 81.0x at (16,64). Exam-cd921bd's
  "Axis 1 green" was a 6/7 claim; RESEARCH.md corrected. run_exam.py now
  covers ccsd_t in phase C (flops --candidate, a new flag) and pins
  candidate sha256 in phase D. Regenerating ccsd_t under v2 would force
  the F-018 fallback (1.0x by construction) but is not claimed.
  REMAINING: none — attribution VERIFIED: per-op profile of the exact
  exam configuration (ccsd_t_selected, (8,32), t1, --allow-risky with the
  exam's prior guarded run as safety evidence) shows output-rank<=1
  einsum ops carry 100.0% of the 21.6 s op-profiled time (top op 2.72 s).
  The reduction-class attribution for the 27.1 s/iter is now profiled,
  not asserted.
- status: CLOSED.

## F-030 F-016's milestone-gated model-v2-on-ADC2 evaluation was due at this milestone and did not happen; the miss is unrecorded [CLOSED — evaluation performed, gap survives v2]
- resolution: accepted; the evaluation ran late-but-in-milestone
  (trials/f030-v2-heldout-eval.jsonl, prediction-only against archived
  measured ranges; no held-out tuning). RESULT: model v2 does NOT close
  the ADC2 gap error — 1.135x predicted vs 1.60–1.92x measured (v1 was
  1.14x; unchanged). The ADC2 underprediction is NOT a reduction-class
  effect; candidate classes are the F-016 k-heavy/low-rank-out gaps.
  CCD deferred question answered: v2 predicts the archived selected 17%
  WORSE than baseline; a8bf146 measured parity — a misrank, recorded with
  the caveat that the measured side is from the unanchored-profile exam
  (refined under F-033's release re-anchor). F-016 remains OPEN.
- symptom: F-016's response and RESEARCH.md open question 4 both state
  model v2 may be evaluated on the held-out ADC2 gap "only at a
  milestone." The exam-cd921bd milestone closed (EXAM done, TOURNAMENT
  recorded, RESEARCH.md updated) with no trial line evaluating v2's
  predicted ADC2 gap against measurement — and none can be reconstructed:
  phase A's adc2 select fell back immediately with predicted_s null, so
  the v2-predicted selected-vs-gristmill ADC2 gap exists nowhere in the
  record. F-016 still rests on v1-era numbers (1.14x predicted vs
  1.60-1.92x measured). Same pattern for CCD: trials/model-v2.jsonl
  explicitly deferred the v2-misranking question ("708 vs 831 vs measured
  ~parity") to "the milestone"; the milestone produced the CCD measurement
  (1002.7 vs 1007.0 ms — parity) and no ledger line evaluates it.

## F-031 Protected exams silently standardize on one of SPEC's three performance sizes [CLOSED — convention documented in SPEC]
- resolution: accepted. SPEC now documents the convention: protected exams
  run performance axes at (16,64) only — the middle size, chosen for cost
  control (the a8bf146/cd921bd exams already take >1 h); (8,40) and (32,96)
  are deferred until a size-sensitivity question is on the table (any exam
  claim involving scaling MUST then re-run all three). Single-size exam
  numbers are labeled single-size in RESEARCH.md.
- symptom: SPEC's performance corpus sizes are {(8,40),(16,64),(32,96)}
  with ccsd_t at (8,32). Both protected exams ran only (16,64). F-015
  forced a documented convention for the Axis-0 (3,5) size; no document
  records why the performance axes use (16,64) alone, or whether (8,40)
  and (32,96) are deferred, dropped, or forgotten. The Axis-2 geomeans
  quoted in RESEARCH.md are single-size claims carrying a corpus-sizes
  label.

## F-032 Post-F-018 Axis 1 is self-certifying on fallback problems; the gate no longer measures compiler quality there [CLOSED — semantics amended: Axis 1 is a safety floor, discrimination is Axis 2]
- symptom: exam phase C ratios of exactly 1.0 for ccd/ccsd/ccsd_lambda/
  eom_ccsd/adc2 compare the emitted verified baseline to itself
  (flop_fallback: true on all five phase-A lines). On those problems the
  gate cannot fail as long as baseline emission works, so "Axis 1 green
  for the first time" reproduces — by construction of the fallback — the
  green-by-construction condition whose earlier assumption-of-green F-018
  itself falsified. RESEARCH.md discloses the identity ("five by
  baseline-fallback identity"), but no entry records that Axis 1 has
  thereby ceased to discriminate the compiler from the baseline on 5/6
  problems, nor that the tournament displacement rule's Axis-1 clause is
  now trivially satisfiable by any challenger that emits the fallback.
- resolution: accepted as a semantics fix, not a bug. F-018 deliberately
  made Axis 1 a SAFETY FLOOR (the compiler never ships worse FLOPs than
  the same-equation baseline) rather than a quality measure; the earlier
  failure was assuming green WITHOUT the cap, not green-by-construction
  with it. What is now recorded: (a) Axis-1 exam reporting must
  distinguish "green by genuine search output" (currently: mp3 only,
  0.969x) from "green by fallback identity" (the other five);
  (b) the contested-slot displacement rule's Axis-1 clause is now a floor
  any fallback-capable challenger satisfies, so displacement
  discrimination rests entirely on the Axis-2 clause; (c) SPEC Axis-1
  text amended to say this. The residual risk the critic names is real:
  a green Axis-1 milestone no longer evidences search quality — that
  evidence now lives only in Axis 2 and in the non-fallback count.

## F-033 All exam wall-time numbers come from a debug-profile binary; rationale and ratio validity unrecorded [CLOSED — ratio validity FALSIFIED for the one genuine win; migration filed as F-036]
- resolution: accepted, and the critic's suspicion was CORRECT. Release
  A/B (same candidates, interleaved, 10/3, load-guarded;
  trials/f027-f033-replication.jsonl): fallback-identity cells hold
  (ccd 0.998/0.983, adc2 0.999/1.086 — matching debug's parity), but mp3
  moves from 1.19x/1.14x (debug) to 1.017-1.045x t1 and
  0.97/1.12/1.12 t16 (release, 3 reps). The release baseline gained more
  from optimization than the selected schedule did (gristmill 1.79x
  debug->release vs selected 1.53x). ALSO: release gristmill mp3 t16 is
  bimodal across processes (50.8 vs 58.0-58.2 ms) — the F-035 tree-roll
  signature in timing data. Consequences recorded: (a) debug-regime
  absolute times and ratios do not transfer to release; (b) SPEC measures
  the compiler's artifact, so release is the honest profile — F-036 owns
  the migration (release calibration + rates refresh + re-exam); (c) the
  exam-cd921bd mp3 win is downgraded to debug-regime-only in RESEARCH.md.
- symptom: the exam line pins binary_profile "debug" (harness sha256
  d761e7972804b49f). a8bf146's unanchored-profile defect is fixed, but the
  choice of a debug build for protected wall-time evaluation is justified
  nowhere. Debug builds change dispatch/permute overhead relative to GEMM
  — precisely the quantities the cost model and the "layout dominates"
  invariant are calibrated on — and whether debug-vs-release preserves the
  measured ratios (including the mp3 win) has never been checked and is
  filed as an open question nowhere. SPEC measures steady-state wall time
  of the compiler's artifact; the exam measured a debug harness.

## F-034 Tournament re-seats the incumbent without comparison for a second consecutive milestone; the only implemented challenger is permanently "not evaluable" [CLOSED — I-008 written; mechanical tournament: challenger emits identical programs]
- resolution: accepted. I-008 (IDEAS.md) gives the structural proposer a
  pre-registered adoption rule. The tournament then ran mechanically:
  structural candidates regenerated under the current engine (model v2 +
  F-018 cap) on {ccd, ccsd, ccsd_lambda, eom_ccsd, adc2}. The proposer DID
  find model-priced improving moves (5-17 accepted steps per problem) but
  every result overshot the Axis-1 cap, so all five emitted candidates are
  program-identical to the incumbent's baseline fallbacks (verified by
  parsed-Program equality). A challenger emitting the identical program
  cannot beat the incumbent on Axis 2: displacement FAILS, incumbent
  stays, no timing runs needed (same program => same execution). mp3: no
  structural action space exists; ccsd_t: F-017-blocked. Recorded in
  trials/tournament-cd921bd.jsonl. The substantive finding: under v2 the
  structural space's improvements are all FLOP-over-cap — the slot's real
  contest now requires beating baseline FLOPs AND wall time together.
- symptom: tournament-cd921bd (like tournament-a8bf146) disposes I-001 and
  the structural proposer as NOT EVALUABLE — the proposer because it has
  never been given an I-entry with a pre-registered adoption rule. That
  bookkeeping gap has now protected the incumbent at two consecutive
  milestones, while the proposer is implemented (benches/candidates/
  *_struct.json exist; it was F-004's exposing case and F-005's failed
  fallback). The mechanical-application rule is being applied to a
  tournament whose challenger roster the project itself left unwritten;
  the slot is uncontested by procedure, not by evidence. No entry
  schedules writing the missing I-entry or records a decision not to.
