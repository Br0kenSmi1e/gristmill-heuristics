# Wall-time comparison: challenger vs the four baseline pipelines

Median wall time in ms at (16,64), single thread / 16 threads. Every cell
gate-verified (elementwise vs the reference program at (3,5)).

Our column: the challenger's candidates on the tenferro backend (release
binary, `--layout model`, median of 10 after 3 warmup, interleaved) —
source: `tools/joint_search/trials/20261105-131316-wallbench.jsonl`.

Baseline columns: gristmill's own schedule through its four shipped code
generators — source: `gristmill-native-baselines/tools/native/RESULTS.md`
(recorded 2026-11-06, same machine).

| problem | **ours** t1/t16 | numpy t1/t16 | C t1/t16 | Fortran t1/t16 | Julia t1/t16 |
|---|---|---|---|---|---|
| `mp3` | **202.6 / 46.3** | 4027.4 / 4168.5 | 29923.5 / 29427.0 | 77584.6 / 68488.4 | 18961.9 / 18731.5 |
| `ccd` | **463.0 / 153.6** | 3178.7 / 3148.2 | 12197.2 / 12889.2 | 74165.0 / 73859.6 | **261.5 / 249.4** |
| `ccsd` | 628.9 / **300.8** | 3851.9 / 3852.9 | 25546.5 / 24557.4 | 85787.4 / 85776.7 | **534.3 / 507.7** |
| `adc2` | **59.4 / 79.2** | 222.1 / 229.0 | 195.8 / 192.1 | 960.6 / 954.3 | 109.2 / 124.6 |
| `eom_ccsd` | 1021.4 / **547.8** | 4130.7 / 4134.7 | 27100.3 / 27613.6 | 112143.1 / 96536.6 | **650.4 / 604.1** |

## Reading

- We beat the numpy, C, and Fortran pipelines everywhere — by up to two
  orders of magnitude on the larger problems.
- The Julia/OMEinsum pipeline is the real competitor: it beats our column
  at t1 on ccd, ccsd, and eom_ccsd. That failure has an address: the
  schedule is fine (see the symbolic scoreboard); our backend's
  single-thread throughput is the next thing to fix.

## Caveats on record

- Protocols differ by column: ours ran median-of-10 after 3 warmup; the
  naive C/Fortran loop printers and the Julia pipeline ran a reduced
  protocol (1 warmup, 3 iterations) because one naive-loop iteration at
  (16,64) takes up to 112 s. The Julia margins are large enough to survive
  the difference; margins under ~2x should be read with that in mind.
- Our column was measured while machine load climbed (3.78 → 7.01 over the
  batch); the late cells (adc2 t16, eom_ccsd) are the most exposed. The
  F-006 load guard was on throughout (refusal above 8).
- The `ccd` t16 cell of ours (153.6) carries that climbing load; treat its
  second decimal as noise.
- `ccsd_t` and `ccsd_lambda` are absent: F-017 blocks full-size execution
  of the former; the latter's wall-time row was never in the shared exam
  corpus.
