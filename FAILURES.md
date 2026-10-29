# FAILURES — the failure registry

One entry per failure: symptom, mechanism (verified BEFORE fix), exposing
fixture, fix, regression pointer. Seeded retroactively from the prequel
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

## F-004 cost model blind to high-rank slow path [OPEN]
- symptom: ccsd_t candidate predicted 0.41 s, measured 8.1 s (20x error);
  model-only acceptance shipped a schedule SLOWER than naive
- mechanism: binary clean/slow GEMM classification too coarse; rank-6
  intermediate-forming ops run at ~0.6 GF/s, 100x below clean GEMM, 37x
  below the modeled slow rate
- planned fix (M1): rank/stride-dependent slow-path rate from a calibration
  grid; model-vs-measured CI gate at 2x; model-only acceptance restricted
  to in-envelope ops
- exposing case: ccsd_t structural-proposer candidate (trials-v1/struct_all.jsonl);
  ADC2 selected-vs-gristmill gap underprediction (F-016)

## F-005 kernel biclique enumeration explodes on large defs [OPEN]
- symptom: ccsd_t def0 query >400 s (kernel enumerates ALL maximal
  bicliques, no budget); selector skips defs >20 terms
- planned fix (M2): enumeration cutoff (greedy_cutoff analogue) + fallback
  to the structural proposer for large defs
- exposing case: ccsd_t via `harness search` (kernel path)

## F-006 benchmark contamination by concurrent load [OPEN — structural fix pending]
- symptom: M3-era batches measured under load avg 24 (~20 julia jobs);
  absolute timings invalid, required full re-measurement
- mechanism: shared machine, no guard; wall-time comparisons silently wrong
- procedural fix in place (PROTOCOL load guard); structural fix (runner
  refuses under load, records load in trial line) not built
- planned fix (M1): load guard inside the timing command itself

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

## F-008 unconditional definition chunking destroys cross-chunk factorization [OPEN — CCD mechanism verified]
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
- planned fix: replace unconditional chunking with the budgeted enumeration
  and structural fallback in F-005; term count alone does not predict the
  ccsd_t explosion, so simply disabling chunking is not a general fix.
- regression: CCD selected-program FLOPs and predicted time no worse than
  the unsplit control; numerical oracle on emitted candidates.

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

## F-018 Axis-1 gate fails on all six measured problems — including the entire train corpus — and no F-entry owns the gate failure [OPEN — gate owner recorded, mechanisms unverified]
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

## F-021 The a8bf146 exam record does not institutionalize the F-006/F-013 lessons: sizes, iteration counts, checksums, and per-benchmark load are all absent [OPEN — structured records added, per-run load pending F-006]
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
  for all 24 runs. Batch-start load is explicitly scoped as such; per-run
  load was not measured and is not fabricated. F-006 retains that structural
  blocker.

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

## F-017 CCSD(T) exam blocked by harness 6 GB limit below the 14 GB regime [OPEN]
- symptom: the a8bf146 protected exam reached ccsd_t gristmill t1 and stopped
  before timing: estimated peak 8587 MB (inputs 137 MB) vs default 6000 MB.
- mechanism: VERIFIED by the runner guard. SPEC allows this machine's 14 GB
  envelope and selects ccsd_t at (8,32) to fit it, but `harness time` defaults
  to a smaller 6000 MB cap. The baseline was not run and no timing exists.
- exposing fixture: `harness time benches/chemistry/ccsd_t.json --sizes
  0=8,1=32 --threads 1 --candidate benches/candidates/ccsd_t_gristmill.json
  --layout model` with default `--max-mem-mb`.
- planned fix: make the exam memory cap an explicit SPEC-consistent setting,
  with actual allocation safety retained; do not silently raise it during an
  exam. The a8bf146 exam is reported incomplete as-is.
