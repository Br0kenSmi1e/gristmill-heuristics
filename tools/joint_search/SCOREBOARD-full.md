# Challenger scoreboards

Branch `joint-search-proto`, worktree `gristmill-joint-search`.
All rows gate-verified (symbolic check + numerical oracle at (3,5)).
Sources: `trials/20261106-110249.jsonl` (main-corpus symbolic table),
`trials/20261109-160027-ccsdt.jsonl`, `trials/20261109-171354-ccsdtq.jsonl`,
`trials/20261109-151107-mpn.jsonl` (derived problems),
`trials/20261105-131316-wallbench.jsonl` (wall table, via SCOREBOARD.md).

## Symbolic operation counts

Leading terms of the symbolic FLOP count (O = occupied, V = virtual;
evaluated at O=1000, V=5000). Runtime columns are the *optimizer's* own
compile time. `ccsd_lambda`/`eom_ccsd`/`ccsd_t` are held-out (never
tuned on). `ccsdt`/`ccsdtq` and the `mpn` rows are problems generated in
this arm (CC residuals via drudge, `gen_cc.py` — validated sig-for-sig
against corpus ccsd before use; `gen_mpn.py` ring diagrams), with
same-equation gristmill baselines produced locally.

| instance | gristmill FLOPs | ILP FLOPs | gristmill runtime | ILP runtime | verified |
|---|---|---|---|---|---|
| `mp3` | 4·O⁴V² + 3·O³V³ + 3·O²V⁴ | **2·O⁴V² + 2·O³V³ + 2·O²V⁴** | 0.50 s | 0.23 s | ✓ |
| `ccd` | 4·O⁴V² + 6·O³V³ + 2·O²V⁴ | 4·O⁴V² + 6·O³V³ + 2·O²V⁴ (tie) | 0.26 s | 0.06 s | ✓ |
| `ccsd` | 8·O⁴V² + 8·O³V³ + 2·O²V⁴ | **6·O⁴V² + 8·O³V³ + 2·O²V⁴** | 1.78 s | 0.31 s | ✓ |
| `adc2` | 16·O³V² + 6·O²V³ | 16·O³V² + 6·O²V³ (exact tie = proof of optimality in the space) | 0.30 s | 0.02 s | ✓ |
| `ccsd_lambda` | 12·O⁴V² + 6·O³V³ + 2·O²V⁴ | **8·O⁴V² + 4·O³V³ + 2·O²V⁴** | 2.34 s | 0.63 s | ✓ |
| `eom_ccsd` | 18·O⁴V² + 10·O³V³ + 2·O²V⁴ | **12·O⁴V² + 10·O³V³ + 2·O²V⁴** | 4.15 s | 0.70 s | ✓ |
| `ccsd_t` | 19·O⁴V³ + 20·O³V⁴ | **4·O⁴V³ + 4·O³V⁴** | 11.83 s | 9.79 s | ✓ |
| `ccsdt` | 14·O⁵V³ + 8·O⁴V⁴ + 2·O³V⁵ | **12·O⁵V³** + 8·O⁴V⁴ + 2·O³V⁵ | 2.90 s | 5.99 s | ✓ |
| `ccsdtq` | 16·O⁶V⁴ + 10·O⁵V⁵ + 2·O⁴V⁶ | **14·O⁶V⁴** + 10·O⁵V⁵ + 2·O⁴V⁶ | 5.70 s | 403 s | ✓ |
| `mpn2` | 6·O²V² | 6·O²V² (tie) | 0.03 s | 0.003 s | ✓ |
| `mpn3` | 4·O³V³ | 4·O³V³ (tie) | 0.08 s | 0.02 s | ✓ |
| `mpn4` | 8·O³V³ | **4·O³V³** (2× win) | 0.15 s | 0.14 s | ✓ |
| `mpn5` | 18·O³V³ | **6·O³V³** (3× win) | 0.27 s | 1.41 s | ✓ |

FLOP ratios at (O=1000, V=5000) for the derived CC rungs: ccsdt 0.9805,
ccsdtq 0.9825 — strict wins on the leading Oᵏ⁺²Vᵏ coefficient (exactly
2 units on each rung). MPn ties low orders; the win grows with order
(2×, 3×) until the factorial-canon wall cuts the series off at n=6
(60 s budget, recorded in the trials line; those rows have no results
and are not shown).

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
- Derived-problem caveats: ccsdtq's ILP runtime is canon-bound (99.9% of
  build in `canon_full`; the MILP itself is subsecond); oracle checks at
  (3,5) take ~17 min per candidate there (dense rank-8 reference eval).
  mpn≥6 hits the same factorial-canon wall inside the 60 s budget;
  gristmill was not run there (its own search also degrades).
