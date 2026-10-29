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
- exposing case: ccsd_t structural-proposer candidate (trials-v1/struct_all.jsonl)

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

## F-007 selector at parity-to-behind vs same-equation gristmill [OPEN]
- symptom: geomean 0.90x t1 across six problems vs gristmill run on the
  IDENTICAL equations (losses: eom_ccsd 0.77x, adc2 0.67x t1; wins: mp3,
  ccsd t16). Earlier "1.42x CCSD win" was a baseline artifact — retracted.
- mechanism: UNKNOWN — next step is a per-op diff of our eom/adc2 schedules
  vs gristmill's to find the moves our rewrite space or policy misses
- data: archive/trials-v1/final_vs_gristmill.jsonl
- NOTE: this entry is also the workflow's first validation — the
  same-equation-baseline discipline caught a false win before publication.
- fix candidates: I-001 (if the diff shows greedy-selection conflicts);
  a new I-entry for extending the rewrite space (if the diff shows
  gristmill makes moves our space lacks). The per-op diff decides.
