#!/usr/bin/env python3
"""Protected milestone exam (PROTOCOL slow loop; SPEC axes 0/1/2).

Runs ONCE per milestone event and is reported as-is — never iterated
against. ccsd_t is expected to refuse at its memory guard (F-017) and is
recorded, not retried.

Phases:
  A. regenerate selected candidates with the current selector (compile
     time; selector enforces exact verify + the Axis-1 cap with verified
     baseline fallback, F-018)
  B. Axis 0: harness check on every candidate/problem pair at (3,5)
     (F-015 convention); any failure stops the exam and voids the pair
  C. Axis 1: FLOP ratio selected/baseline from the select records
  D. Axis 2: same-backend diagnostic, interleaved A/B, (16,64), t1/t16,
     10 iters / 3 warmup; the harness load guard (F-006) is structural

Outputs: trials/exam-<git>-runs.jsonl (one record per run, load fields
included per run) and a summary line in trials/exam-<git>.jsonl.
"""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Binary profile: RELEASE since model v3 (F-036) — SPEC measures the
# compiler's artifact, and v3 rates are release-calibrated. (a8bf146 never
# recorded its profile; cd921bd pinned debug.)
HARNESS = ROOT / "target" / "release" / "harness"
TRIALS = ROOT / "trials"

# Corpus order: train first, then held-out. ccsd_t last; its memory-guard
# refusal is expected (F-017) and recorded, not retried.
PROBLEMS = ["mp3", "ccd", "ccsd", "ccsd_lambda", "eom_ccsd", "adc2", "ccsd_t"]
SELECT_SIZES = "0=16,1=64"
CHECK_SIZES = "0=3,1=5"
TIME_SIZES = {"ccsd_t": "0=8,1=32"}  # SPEC: ccsd_t fits the 14 GB envelope at (8,32)
DEFAULT_TIME_SIZES = "0=16,1=64"
SELECT_TIMEOUT_S = 7200
TIMEOUT_S = 900


def git_hash():
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def loadavg():
    return [float(x) for x in Path("/proc/loadavg").read_text().split()[:3]]


def run(cmd, timeout):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                           timeout=timeout)
        rc, out, err_tail = p.returncode, p.stdout, p.stderr.strip().splitlines()[-3:]
    except subprocess.TimeoutExpired as e:
        # Never die with a bare traceback inside a protected exam: record it.
        rc, out = "timeout", (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        err_tail = [f"TIMEOUT after {timeout}s"]
    rec = {"cmd": " ".join(str(c) for c in cmd), "exit": rc,
           "wall_s": round(time.time() - t0, 2)}
    lines = [l for l in out.splitlines() if l.strip().startswith("{")]
    if lines:
        rec["json"] = json.loads(lines[-1])
    if rc != 0:
        rec["stderr_tail"] = err_tail
    return rec


def main():
    git = git_hash()
    runs_path = TRIALS / f"exam-{git}-runs.jsonl"
    exam_path = TRIALS / f"exam-{git}.jsonl"
    runs = runs_path.open("a")
    summary = {"git": git, "kind": "protected-exam", "load_at_start": loadavg(),
               "harness": str(HARNESS), "binary_profile": "release",
               "harness_sha256": sha256(HARNESS), "phases": {}}
    print(f"exam start: git {git}, load {summary['load_at_start']}, runs -> {runs_path}")

    # ---- Phase A: regenerate selected candidates (compile time) ----
    select_recs = {}
    for prob in PROBLEMS:
        if prob == "ccsd_t":
            continue  # selector output for ccsd_t is not needed: timing blocked (F-017)
        out = ROOT / "benches" / "candidates" / f"{prob}_selected.json"
        cmd = [HARNESS, "select", f"benches/chemistry/{prob}.json",
               "--sizes", SELECT_SIZES, "--threads", "1", "--out", out]
        rec = run(cmd, SELECT_TIMEOUT_S)
        rec["phase"] = "A-select"
        rec["problem"] = prob
        if out.exists():
            rec["candidate_sha256"] = sha256(out)
        runs.write(json.dumps(rec) + "\n")
        runs.flush()
        select_recs[prob] = rec
        j = rec.get("json", {})
        print(f"  A {prob}: exit={rec['exit']} fallback={j.get('flop_fallback')} "
              f"steps={len(j.get('trace', []))} wall={rec['wall_s']}s")
        if rec["exit"] != 0:
            summary["phases"]["A"] = {"failed_on": prob, "record": rec}
            finish(summary, exam_path)
            sys.exit(f"phase A failed on {prob}; exam void, see {runs_path}")
    summary["phases"]["A"] = "ok"

    # ---- Phase B: Axis 0 (gate) ----
    pairs = []
    for prob in PROBLEMS:
        for cand in [f"{prob}_selected", f"{prob}_gristmill"]:
            pairs.append((prob, cand))
    worst = 0.0
    for prob, cand in pairs:
        cmd = [HARNESS, "check", f"benches/chemistry/{prob}.json",
               "--sizes", CHECK_SIZES, "--candidate",
               f"benches/candidates/{cand}.json"]
        rec = run(cmd, TIMEOUT_S)
        rec["phase"] = "B-axis0"
        rec["problem"] = prob
        rec["candidate"] = cand
        runs.write(json.dumps(rec) + "\n")
        runs.flush()
        ok = rec["exit"] == 0 and rec.get("json", {}).get("pass")
        wr = rec.get("json", {}).get("worst_rel_diff", float("nan"))
        worst = max(worst, wr if wr == wr else 0.0)
        print(f"  B {prob}/{cand}: pass={ok} worst_rel={wr:.2e}")
        if not ok:
            summary["phases"]["B"] = {"failed_on": f"{prob}/{cand}", "record": rec}
            finish(summary, exam_path)
            sys.exit(f"Axis 0 failed on {prob}/{cand}; exam void, see {runs_path}")
    summary["phases"]["B"] = {"pass": True, "pairs": len(pairs), "worst_rel_diff": worst}

    # ---- Phase C: Axis 1 (gate, from phase-A records) ----
    flop_ratios = {}
    axis1_fail = []
    for prob, rec in select_recs.items():
        j = rec.get("json", {})
        if "log10_flops" not in j or "flop_cap_log10" not in j:
            axis1_fail.append(prob)
            continue
        # cap = baseline + ln(1.05); ratio vs baseline:
        ratio = 10 ** (j["log10_flops"] - (j["flop_cap_log10"] - json_log10_105()))
        flop_ratios[prob] = round(ratio, 4)
        if ratio > 1.05 + 1e-9:
            axis1_fail.append(prob)
        print(f"  C {prob}: flop_ratio={flop_ratios[prob]} fallback={j.get('flop_fallback')}")
    # ccsd_t: execution-free Axis-1 via flops --candidate (F-029; timing is
    # F-017-blocked but the FLOP gate is not).
    ccsd_t_flops = {}
    for cand in ["ccsd_t_selected", "ccsd_t_gristmill"]:
        rec = run([HARNESS, "flops", "benches/chemistry/ccsd_t.json",
                   "--sizes", TIME_SIZES["ccsd_t"], "--candidate",
                   f"benches/candidates/{cand}.json"], TIMEOUT_S)
        rec["phase"] = "C-axis1-ccsd_t"
        rec["problem"] = "ccsd_t"
        rec["candidate"] = cand
        rec["candidate_sha256"] = sha256(ROOT / "benches" / "candidates" / f"{cand}.json")
        runs.write(json.dumps(rec) + "\n")
        runs.flush()
        ccsd_t_flops[cand] = rec.get("json", {}).get("log10_flops")
    if all(v is not None for v in ccsd_t_flops.values()):
        ratio = 10 ** (ccsd_t_flops["ccsd_t_selected"] - ccsd_t_flops["ccsd_t_gristmill"])
        flop_ratios["ccsd_t"] = round(ratio, 4)
        if ratio > 1.05 + 1e-9:
            axis1_fail.append("ccsd_t")
        print(f"  C ccsd_t: flop_ratio={flop_ratios['ccsd_t']} (stale candidate, execution-free)")
    summary["phases"]["C"] = {"flop_ratios": flop_ratios, "axis1_failures": axis1_fail}

    # ---- Phase D: Axis 2 same-backend diagnostic, interleaved A/B ----
    timings = {}
    cell = 0
    for prob in PROBLEMS:
        sizes = TIME_SIZES.get(prob, DEFAULT_TIME_SIZES)
        for threads in [1, 16]:
            cands = [f"{prob}_selected", f"{prob}_gristmill"]
            if cell % 2 == 1:
                cands.reverse()  # alternate A/B order across cells
            cell += 1
            for cand in cands:
                cmd = [HARNESS, "time", f"benches/chemistry/{prob}.json",
                       "--sizes", sizes, "--threads", str(threads),
                       "--candidate", f"benches/candidates/{cand}.json",
                       "--layout", "model", "--iters", "10", "--warmup", "3"]
                rec = run(cmd, TIMEOUT_S)
                rec["phase"] = "D-axis2"
                rec["problem"] = prob
                rec["candidate"] = cand
                rec["threads"] = threads
                rec["candidate_sha256"] = sha256(
                    ROOT / "benches" / "candidates" / f"{cand}.json")
                runs.write(json.dumps(rec) + "\n")
                runs.flush()
                j = rec.get("json", {})
                med = j.get("median_ms")
                print(f"  D {prob}/{cand} t{threads}: exit={rec['exit']} "
                      f"median={med and round(med, 2)}ms load={j.get('load')}")
                timings.setdefault(prob, {})[f"t{threads}_{cand.split('_', 1)[1]}"] = {
                    "exit": rec["exit"], "median_ms": med,
                    "refused": rec["exit"] != 0,
                }
    summary["phases"]["D"] = timings
    finish(summary, exam_path)
    print(f"exam complete: {exam_path}")


def json_log10_105():
    import math
    return math.log10(1.05)


def finish(summary, exam_path):
    summary["load_at_end"] = loadavg()
    with exam_path.open("a") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"summary -> {exam_path}")


if __name__ == "__main__":
    main()
