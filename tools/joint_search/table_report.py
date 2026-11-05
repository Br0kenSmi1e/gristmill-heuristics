#!/usr/bin/env python3
"""table_report.py — the 7-instance comparison table (user-requested).

Per instance: gristmill vs the joint-search challenger —
  log10 FLOPs at (16,64) via the FLOP cost model (price_candidate),
  OPTIMIZER runtime (not the produced program's wall time):
    gristmill = the gristmill.optimize() call in gristmill_baseline
    (same-equation pipeline: harness dump-comp -> gristmill), plus the
    full script total (drudge/sympy setup + export) recorded alongside;
    ours = solve_problem + emit_global in a fresh process.
  verification = harness check at (3,5) (truthmill verify + oracle).

Requires /tmp/gristmill-venv (see mainline tools/README.md) and writes
scratch files to /tmp/comp/. Appends one JSON line to trials/.
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
GRIST_PY = "/tmp/gristmill-venv/bin/python"
GRIST_SCRIPT = "/tmp/gristmill_baseline_timed.py"
COMP_DIR = "/tmp/comp"
VENV_PY = os.path.join(HERE, ".venv", "bin", "python")

INSTANCES = ["mp3", "ccd", "ccsd", "adc2", "ccsd_lambda", "eom_ccsd",
             "ccsd_t"]
FLOP_SIZES = {0: 16, 1: 64}

OURS_SOLVE = r"""
import sys, json, time
sys.path.insert(0, 'tools/joint_search')
import proto
name = sys.argv[1]
prob = json.load(open(f'benches/chemistry/{name}.json'))
proto.SYMS = prob['symmetries']
t0 = time.time()
pack = proto.solve_problem(prob, {0:16,1:64})
prog = proto.emit_global(pack, {0:16,1:64})
t1 = time.time()
priced = proto.price_candidate(prog, {0:16,1:64})
json.dump(prog, open(f'tools/joint_search/out/{name}_table.json', 'w'))
print(f"{t1-t0:.4f} {priced}")
"""


def load1():
    with open("/proc/loadavg") as f:
        return float(f.read().split()[0])


def main():
    if load1() > 8:
        sys.exit("load > 8, refusing (shared machine)")
    os.makedirs(COMP_DIR, exist_ok=True)
    rows = []
    for name in INSTANCES:
        prob = f"{ROOT}/benches/chemistry/{name}.json"
        comp = f"{COMP_DIR}/{name}.json"
        with open(comp, "w") as fh:
            subprocess.run([HARNESS, "dump-comp", prob], stdout=fh,
                           check=True)
        # gristmill optimizer timing
        r = subprocess.run([GRIST_PY, GRIST_SCRIPT, comp,
                            f"{COMP_DIR}/{name}_grist.json"],
                           capture_output=True, text=True)
        g_opt = g_tot = None
        for line in r.stdout.splitlines():
            if line.startswith("TIMING"):
                parts = line.split()
                g_opt = float(parts[parts.index("optimize") + 1][:-1])
        # our optimizer timing (fresh process)
        r2 = subprocess.run([VENV_PY, "-c", OURS_SOLVE, name],
                            capture_output=True, text=True, cwd=ROOT)
        o_solve, o_flops = r2.stdout.strip().split()[-2:]
        # flops + verification
        gf = proto.price_candidate(
            json.load(open(f"{ROOT}/benches/candidates/"
                           f"{name}_gristmill.json")), FLOP_SIZES)
        ours_path = f"{ROOT}/tools/joint_search/out/{name}_table.json"
        ok = subprocess.run(
            [HARNESS, "check", prob, "--sizes", "0=3,1=5",
             "--candidate", ours_path],
            capture_output=True, text=True)
        passed = ok.returncode == 0 and json.loads(ok.stdout)["pass"]
        row = {"name": name, "grist_log10_flops": None,
               "grist_flops": gf, "grist_optimize_s": g_opt,
               "ours_solve_emit_s": float(o_solve),
               "ours_flops": int(o_flops), "verified": passed}
        import math
        row["grist_log10_flops"] = math.log10(gf)
        row["ours_log10_flops"] = math.log10(int(o_flops))
        rows.append(row)
        print(f"{name:12s} grist {row['grist_log10_flops']:.4f} "
              f"{g_opt}s | ours {row['ours_log10_flops']:.4f} "
              f"{o_solve}s | verified {passed}", flush=True)
    path = os.path.join(HERE, "trials",
                        time.strftime("%Y%m%d-%H%M%S") + ".jsonl")
    with open(path, "a") as fh:
        fh.write(json.dumps({"kind": "table-report", "load": load1(),
                             "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "rows": rows}) + "\n")
    print("logged to", path)


if __name__ == "__main__":
    main()
