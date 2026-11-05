#!/usr/bin/env python3
"""Heldout evaluation: solve + emit + verify + price problems OUTSIDE the
train set, against their same-equation gristmill baselines. Read-only
measurement — no tuning on these problems (protected sets).

Usage (from worktree root):
  tools/joint_search/.venv/bin/python tools/joint_search/eval_heldout.py NAME...
Appends one JSON line to trials/.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import proto  # noqa: E402

HARNESS = os.path.join(ROOT, "target", "debug", "harness")
TRIALS = os.path.join(HERE, "trials")
OUT = os.path.join(HERE, "out")


def load1():
    with open("/proc/loadavg") as f:
        return float(f.read().split()[0])


def eval_one(name):
    ppath = f"{ROOT}/benches/chemistry/{name}.json"
    cpath = f"{ROOT}/benches/candidates/{name}_gristmill.json"
    problem = json.load(open(ppath))
    proto.SYMS = problem["symmetries"]
    sizes = {0: 16, 1: 64}
    t0 = time.time()
    pack = proto.solve_problem(problem, sizes)
    prog = proto.emit_global(pack, sizes)
    solve_s = time.time() - t0
    opath = os.path.join(OUT, f"{name}_heldout.json")
    json.dump(prog, open(opath, "w"))
    priced = proto.price_candidate(prog, sizes)
    priced_wall = proto.wallmodel.price_candidate_wall(
        prog, sizes, proto.wall_rates())
    r = subprocess.run([HARNESS, "check", ppath, "--sizes", "0=3,1=5",
                        "--candidate", opath],
                       capture_output=True, text=True)
    check = ({"pass": json.loads(r.stdout)["pass"],
              "worst_rel_diff": json.loads(r.stdout)["worst_rel_diff"]}
             if r.returncode == 0 else
             {"pass": False, "error": (r.stderr or r.stdout).strip()[:300]})
    base = proto.price_candidate(json.load(open(cpath)), sizes)
    base_wall = proto.wallmodel.price_candidate_wall(
        json.load(open(cpath)), sizes, proto.wall_rates())
    return {"tag": f"{name}_heldout", "ilp_objective_wall_s": pack["obj"],
            "emitted_priced": priced, "gristmill_flops": base,
            "emitted_wall_s": priced_wall, "gristmill_wall_s": base_wall,
            "ratio": priced / base,
            "wall_ratio_model": priced_wall / base_wall,
            "check": check, "solve_s": round(solve_s, 2),
            "stats": pack["stats"]}


def main():
    names = sys.argv[1:]
    load = load1()
    if load > 8:
        print(f"load {load} > 8, refusing (shared machine)")
        sys.exit(1)
    rec = {"kind": "heldout-eval", "load": load,
           "time": time.strftime("%Y-%m-%d %H:%M:%S"), "results": []}
    for name in names:
        try:
            r = eval_one(name)
        except Exception as e:
            r = {"tag": f"{name}_heldout", "error": f"{type(e).__name__}: {e}"}
        rec["results"].append(r)
        if "error" in r:
            print(f"{name}: ERROR {r['error']}")
        else:
            print(f"{name}: flop-ratio {r['ratio']:.6f} "
                  f"wall-ratio(model) {r['wall_ratio_model']:.4f} "
                  f"pass {r['check']['pass']} solve {r['solve_s']}s "
                  f"vars {r['stats']['vars']}")
    path = os.path.join(TRIALS, time.strftime("%Y%m%d-%H%M%S") + ".jsonl")
    with open(path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print("logged to", path)


if __name__ == "__main__":
    main()
