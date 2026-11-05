#!/usr/bin/env python3
"""The challenger-arm battery: solve + emit + verify + price the train
problems and fixtures, append one JSON line to trials/.

Usage (from the worktree root):
  /path/to/venv/bin/python tools/joint_search/run_battery.py [label]

Gates per LOOP.md: emission self-check (objective vs priced), harness
check (truthmill validate + numerical oracle at (3,5)), budget check.
Never run under load > 8 (shared machine; the mainline times things).
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
os.makedirs(TRIALS, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

TRAIN = ["mp3", "ccd", "ccsd", "adc2"]
FIXTURES = ["antisym_test", "antisym_test2", "ccsd_energy",
            "ccsd_doubles_38_39_40_42_43", "ccsd_doubles_36_37_50_51"]


def load1():
    with open("/proc/loadavg") as f:
        return float(f.read().split()[0])


def git_head():
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                   cwd=ROOT).decode().strip()


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def harness_check(problem_json, cand_json):
    r = run([HARNESS, "check", problem_json, "--sizes", "0=3,1=5",
             "--candidate", cand_json])
    if r.returncode != 0:
        return {"pass": False, "error": (r.stderr or r.stdout).strip()[:200]}
    d = json.loads(r.stdout)
    return {"pass": d["pass"], "worst_rel_diff": d["worst_rel_diff"]}


def one(problem_path, tag, sizes, sym):
    """Solve + emit + price + check one problem. Returns a record."""
    t0 = time.time()
    problem = json.load(open(problem_path))
    if sym:
        proto.SYMS = problem["symmetries"]
    else:
        proto.SYMS = {}
    pack = proto.solve_problem(problem, sizes)
    prog = proto.emit_global(pack, sizes)
    cand_path = os.path.join(OUT, f"{tag}.json")
    json.dump(prog, open(cand_path, "w"))
    priced = proto.price_candidate(prog, sizes)
    priced_wall = proto.wallmodel.price_candidate_wall(
        prog, sizes, proto.wall_rates())
    check = harness_check(problem_path, cand_path)
    return {"tag": tag, "ilp_objective_wall_s": pack["obj"],
            "ilp_objective_flops": pack["stats"]["flops_at_solution"],
            "emitted_priced": priced, "emitted_wall_s": priced_wall,
            "wall_check_delta_s": priced_wall - pack["obj"],
            "check": check, "solve_s": round(time.time() - t0, 2)}


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else ""
    load = load1()
    if load > 8:
        print(f"load {load} > 8, refusing (shared machine)")
        sys.exit(1)

    venv_py = os.path.join(HERE, ".venv", "bin", "python")
    run([venv_py if os.path.exists(venv_py) else sys.executable,
        os.path.join(HERE, "make_fixtures.py")], cwd=ROOT)

    record = {"git": git_head(), "note": label, "load": load,
              "kind": "battery", "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "results": []}

    for p in TRAIN:
        for sym in (False, True):
            tag = f"{p}_global{'_sym' if sym else ''}"
            rec = one(f"{ROOT}/benches/chemistry/{p}.json", tag,
                      {0: 16, 1: 64}, sym)
            base = proto.price_candidate(
                json.load(open(f"{ROOT}/benches/candidates/"
                               f"{p}_gristmill.json")), {0: 16, 1: 64})
            rec["gristmill_flops"] = base
            rec["ratio"] = rec["emitted_priced"] / base
            rec["gristmill_wall_s"] = proto.wallmodel.price_candidate_wall(
                json.load(open(f"{ROOT}/benches/candidates/"
                               f"{p}_gristmill.json")), {0: 16, 1: 64},
                proto.wall_rates())
            rec["wall_ratio_model"] = rec["emitted_wall_s"] / rec["gristmill_wall_s"]
            record["results"].append(rec)
            print(f"{tag}: flop-ratio {rec['ratio']:.4f} "
                  f"wall-ratio(model) {rec['wall_ratio_model']:.4f} "
                  f"pass {rec['check']['pass']}")

    for f in FIXTURES:
        rec = one(os.path.join(HERE, "fixtures", f + ".json"),
                  f, {0: 16, 1: 64}, True)
        record["results"].append(rec)
        print(f"{f}: pass {rec['check']['pass']}")

    line = json.dumps(record)
    path = os.path.join(TRIALS,
                        time.strftime("%Y%m%d-%H%M%S") + ".jsonl")
    with open(path, "a") as fh:
        fh.write(line + "\n")
    print("logged to", path)

    fails = [r["tag"] for r in record["results"]
             if not r["check"]["pass"]]
    print("FAILURES:" if fails else "all pass",
          ", ".join(fails) if fails else "")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
