# Challenger scoreboards

Branch `joint-search-proto`, worktree `gristmill-joint-search`.
All rows gate-verified (symbolic check + numerical oracle at (3,5)).
Sources: `trials/20261106-110249.jsonl` (symbolic table),
`trials/20261105-131316-wallbench.jsonl` (wall table, via SCOREBOARD.md).

## Symbolic operation counts

Leading terms of the symbolic FLOP count (O = occupied, V = virtual;
evaluated at O=1000, V=5000). Runtime columns are the *optimizer's* own
compile time. Last three rows are held-out (never tuned on).

| instance | baseline FLOPs | our FLOPs | baseline runtime | our runtime | verified |
|---|---|---|---|---|---|
| `mp3` | 4·O⁴V² + 3·O³V³ + 3·O²V⁴ | **2·O⁴V² + 2·O³V³ + 2·O²V⁴** | 0.50 s | 0.23 s | ✓ |
| `ccd` | 4·O⁴V² + 6·O³V³ + 2·O²V⁴ | 4·O⁴V² + 6·O³V³ + 2·O²V⁴ (tie) | 0.26 s | 0.06 s | ✓ |
| `ccsd` | 8·O⁴V² + 8·O³V³ + 2·O²V⁴ | **6·O⁴V² + 8·O³V³ + 2·O²V⁴** | 1.78 s | 0.31 s | ✓ |
| `adc2` | 16·O³V² + 6·O²V³ | 16·O³V² + 6·O²V³ (exact tie = proof of optimality in the space) | 0.30 s | 0.02 s | ✓ |
| `ccsd_lambda` | 12·O⁴V² + 6·O³V³ + 2·O²V⁴ | **8·O⁴V² + 4·O³V³ + 2·O²V⁴** | 2.34 s | 0.63 s | ✓ |
| `eom_ccsd` | 18·O⁴V² + 10·O³V³ + 2·O²V⁴ | **12·O⁴V² + 10·O³V³ + 2·O²V⁴** | 4.15 s | 0.70 s | ✓ |
| `ccsd_t` | 19·O⁴V³ + 20·O³V⁴ | **4·O⁴V³ + 4·O³V⁴** | 11.83 s | 9.79 s | ✓ |

## Wall time (measured, not model)

Median wall time ratio ours/gristmill (same-equation candidates), at
(16,64), release binary, `--layout model`, median of 10 after 3 warmup,
interleaved A/B, per-run load recorded. Ratio < 1.0 = we are faster.

| problem | t1 | t16 | model predicted |
|---|---|---|---|
| `mp3` | **0.818** | **0.817** | 0.929 |
| `ccd` | **0.636** | 1.063 | 0.954 |
| `ccsd` | **0.880** | **0.852** | 0.933 |
| `adc2` | **0.621** | 1.019 | 1.001 |
| `eom_ccsd` (held-out) | 1.137 | 1.303 | — |

## Caveats on record

- The wall table predates this morning's symbolic-table state (the arm
  has since improved ccd/ccsd in FLOPs); refresh via `run_wallbench.py`
  at the next milestone.
- The eom_ccsd wall loss was measured while machine load was climbing;
  direction looks real, magnitude uncertain (recorded in the trial line).
- Model-vs-measured honesty rule (F-004): the model predicted small
  wins/parity; measured t1 wins are larger (adc2: 1.001 model → 0.621
  measured). No model tuning was done against these numbers.
- adc2's exact tie is a *result*: the ILP is exact (gap zero), so within
  the currently encoded space, nothing cheaper than the baseline exists.
