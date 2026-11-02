# gristmill-heuristics → the tenferro compiler project

A compiler from symbolic tensor equations (truthmill Problem IR) to
verified, memory-safe, wall-time-optimized tenferro code, built by a
failure-driven agent workflow.

Start here:

- `PROTOCOL.md` — how work happens (read first, every session)
- `SPEC.md` — what the compiler must do, with executable acceptance axes
- `FAILURES.md` — the failure registry (what broke, why, fixed or open)
- `IDEAS.md` — the algorithm ledger (hypotheses with pre-registered
  adoption rules)
- `RESEARCH.md` — current understanding, compressed
- `trials/` — append-only experiment log

Layout:

- `crates/harness` — compiler prototype + environment (verify, oracle,
  timing, search, cost model). To be split into product vs environment
  crates (M1).
- `benches/chemistry` — problem corpus (train: mp3, ccd, ccsd; held-out:
  ccsd_t, eom_ccsd, adc2, ccsd_lambda)
- `benches/candidates` — imported baseline schedules (same-equation
  gristmill exports) and selector outputs
- `tools/` — baseline import/export scripts
- `profiles/` — pinned baseline anchor profiles (per-op measurements of the
  gristmill candidates, one frozen set: 6 problems x (16,64) x {t1,t16};
  ccsd_t absent, blocked by F-017). `harness report` is the analysis
  panel: per-op timings vs model, anchor comparison, validity footer.
- `archive/` — the prequel heuristic-learning study (tag `hl-study-final`),
  kept for audit