#!/usr/bin/env python3
"""run_wallbench.py — Part-2 wall-time benchmark (milestone-authorized).

Per-instance wall-time ratio ours/gristmill at t1 and t16, (16,64),
release binary, --layout model, median of 10 after 3 warmup, interleaved
A/B, per-run load recorded by the harness (F-006 guard built in; never
--allow-loaded). Appends one JSONL line per run plus a summary line to
tools/joint_search/trials/.

Usage: run_wallbench.py <ours.json> <problem> [<ours.json> <problem> ...]
"""

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
HARNESS = os.path.join(ROOT, "target", "release", "harness")
TRIALS = os.path.join(HERE, "trials")


def load1():
    with open("/proc/loadavg") as f:
        return float(f.read().split()[0])


def git_head():
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                   cwd=ROOT).decode().strip()


def main():
    pairs = [(sys.argv[i], sys.argv[i + 1])
             for i in range(1, len(sys.argv), 2)]
    load = load1()
    print(f"wallbench start: load {load}, git {git_head()}")
    if load > 8:
        sys.exit(f"load {load} > 8, refusing (load guard)")
    runs = []
    cell = 0
    for ours, prob in pairs:
        base = os.path.join(ROOT, "benches", "candidates",
                            f"{prob}_gristmill.json")
        for threads in ("1", "16"):
            arms = [ours, base]
            if cell % 2 == 1:
                arms.reverse()
            cell += 1
            for cand in arms:
                cmd = [HARNESS, "time",
                       os.path.join(ROOT, "benches", "chemistry",
                                    f"{prob}.json"),
                       "--sizes", "0=16,1=64", "--threads", threads,
                       "--candidate", cand, "--layout", "model",
                       "--iters", "10", "--warmup", "3"]
                t0 = time.time()
                r = subprocess.run(cmd, capture_output=True, text=True,
                                   cwd=ROOT, timeout=1200)
                rec = {"kind": "wallbench-run", "problem": prob,
                       "candidate": os.path.basename(cand),
                       "threads": int(threads), "exit": r.returncode,
                       "wall_s": round(time.time() - t0, 2)}
                lines = [l for l in r.stdout.splitlines()
                         if l.strip().startswith("{")]
                if lines:
                    rec.update(json.loads(lines[-1]))
                if r.returncode != 0:
                    rec["stderr_tail"] = r.stderr.strip().splitlines()[-2:]
                print(f"  {prob} {os.path.basename(cand)} t{threads}: "
                      f"exit {r.returncode} "
                      f"median {rec.get('median_ms')} ms "
                      f"load {rec.get('load')}")
                runs.append(rec)
    # summary
    summary = {"kind": "wallbench", "git": git_head(), "load_start": load,
               "load_end": load1(), "binary": "release",
               "sizes": {"0": 16, "1": 64}, "layout": "model",
               "iters": 10, "warmup": 3, "time": time.strftime("%F %T"),
               "runs": runs, "ratios": {}}
    for ours, prob in pairs:
        entry = {}
        ours_stem = os.path.basename(ours).replace(".json", "")
        for t in (1, 16):
            ours_ms = [r["median_ms"] for r in runs
                       if r["problem"] == prob and r["threads"] == t
                       and r["candidate"] == ours_stem]
            base_ms = [r["median_ms"] for r in runs
                       if r["problem"] == prob and r["threads"] == t
                       and r["candidate"] == f"{prob}_gristmill.json"]
            if ours_ms and base_ms and ours_ms[0] and base_ms[0]:
                entry[f"t{t}"] = {
                    "ours_ms": ours_ms[0], "gristmill_ms": base_ms[0],
                    "ratio_ours_over_gristmill": ours_ms[0] / base_ms[0],
                    "speedup_gristmill_over_ours": base_ms[0] / ours_ms[0],
                }
        summary["ratios"][prob] = entry
    path = os.path.join(TRIALS, time.strftime("%Y%m%d-%H%M%S")
                        + "-wallbench.jsonl")
    with open(path, "a") as fh:
        fh.write(json.dumps(summary) + "\n")
    print("logged to", path)
    print(json.dumps(summary["ratios"], indent=2))


if __name__ == "__main__":
    main()
