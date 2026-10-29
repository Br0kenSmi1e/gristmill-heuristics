# PROTOCOL — how work happens in this repo

Read this first, every session. Then RESEARCH.md. Then pick work from
FAILURES.md (fast loop) or run the milestone protocol (slow loop).

## The fast loop (one iteration = hours)

Work item = close one open F-entry, price one IDEAS.md entry, or close one
named spec gap. Never "improve the metric" in the abstract.

1. SELECT — the highest-priority open F-entry, or the next scheduled item.
2. REPRODUCE — pin a minimal failing test/fixture FIRST. No fixture, no fix.
3. DIAGNOSE — verify the mechanism (profile, trace, per-op diff) and record
   it in the F-entry BEFORE attempting the fix.
4. FIX — one targeted change.
5. GATE, cheapest first, all must pass:
   a. `cargo test` (seconds)
   b. verify + numerical oracle on touched problems: `harness check ...`
   c. model-vs-measured spot check if the cost model was touched
   d. benchmark spot-check, ONLY under the load guard (see below)
6. LOG — one JSON line in `trials/`: git hash, F/I-entries touched, metrics,
   machine load at start.
7. COMPRESS — every ~10 iterations: fold closed F-entries into RESEARCH.md
   invariants; refactor if fix patterns revealed structural debt.

## The slow loop (milestone boundary = days–weeks)

A milestone closes when, in order:

1. EXAM — the protected evaluation (SPEC.md axes) is run ONCE on a clean
   machine and reported as-is. It is never iterated against.
2. TOURNAMENT — contested algorithm slots (IDEAS.md entries marked
   `contested`) are re-compared under the current calibrated cost model,
   with measured spot-checks for the top candidates. Adoption rules written
   in the idea entries are applied MECHANICALLY — no discretion, no "but the
   new one feels promising."
3. CRITIC — a fresh-context reviewer reads RESEARCH.md, FAILURES.md,
   IDEAS.md, and the milestone trial records, and files challenges as new
   F-entries. The milestone stays OPEN until every challenge is answered
   (fixed, or rebutted with evidence).

## Standing rules

- **Mechanism before fix.** A failure whose "why" is not verified is not
  ready to be fixed.
- **Same-equation baselines only.** All comparisons against gristmill use
  `tools/gristmill_baseline.py` exports from the identical truthmill
  references. (Learned the hard way: a weaker baseline path once fabricated
  a 1.42x "win". See archive/RESEARCH-v1.md, correction note.)
- **Load guard.** No benchmark runs when the machine is loaded; check
  `uptime` / running jobs first; interleave A/B; median of >=10 after >=3
  warmup; record load in the trial line. A contaminated batch is worse than
  no batch (see F-006).
- **Model-only acceptance is screening, not shipping.** The cost model may
  rank and kill candidates; anything that lands in a released schedule gets
  a measured confirmation. (F-004.)
- **Protected sets.** Develop on {mp3, ccd, ccsd}; never tune against
  {ccsd_t, eom_ccsd, adc2, ccsd_lambda}; the drop-in test problem (CC2) is
  touched only at its milestone.
- **Exploration budget.** ~20% of iterations go to IDEAS.md entries priced
  below their cheap-test threshold. Protected from engineering pressure.
- **Scope rule.** Building anything SPEC.md Layer 0 concedes (Wick
  expansion, our own factorization kernel, a new backend) must first be
  filed as an F-entry against the spec.
- **Stop rule.** Two consecutive milestones with zero new F-entries on the
  dev corpus => the failure tail is exhausted; run the exam, declare done
  or pivot. Do not manufacture work.

## Roles

- Main context: owns the ledgers, selects work, interprets evidence, talks
  to the user. Minimal implementation.
- Implementation contexts: one task card each ("close F-007, fixture here,
  gate command here"). Quarantined from ledger decisions.
- Critic: fresh context, milestone boundary, must file challenges. Has
  teeth: challenges gate milestone closure.
