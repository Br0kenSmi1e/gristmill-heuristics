# trials/ — append-only experiment log

One JSON object per line, written by the harness/scripts (never edited by
hand; superseded lines stay). Every line carries:

- `git`: commit hash at run time
- `note`: one sentence — what changed and why
- `f` / `i`: F-entries / I-entries touched (empty list allowed)
- `load`: machine load average at start (the F-006 guard)
- metrics: command-dependent (median_ms, threads, sizes, problem, ...)

Prequel logs live in `archive/trials-v1/` (see tag `hl-study-final`).
