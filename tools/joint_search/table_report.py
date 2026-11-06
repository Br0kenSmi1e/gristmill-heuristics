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

import math

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import proto  # noqa: E402

HARNESS = os.path.join(ROOT, "target", "debug", "harness")
GRIST_PY = "/tmp/gristmill-venv/bin/python"
GRIST_SCRIPT = "/tmp/gristmill_baseline_timed5.py"  # substs nv=5*no <=> O=1000,V=5000
COMP_DIR = "/tmp/comp"
VENV_PY = os.path.join(HERE, ".venv", "bin", "python")

INSTANCES = ["mp3", "ccd", "ccsd", "adc2", "ccsd_lambda", "eom_ccsd",
             "ccsd_t"]
FLOP_SIZES = {0: 1000, 1: 5000}

OURS_SOLVE = r"""
import sys, json, time
sys.path.insert(0, 'tools/joint_search')
import proto
name = sys.argv[1]
prob = json.load(open(f'benches/chemistry/{name}.json'))
proto.SYMS = prob['symmetries']
t0 = time.time()
pack = proto.solve_problem(prob, {0:1000,1:5000})
prog = proto.emit_global(pack, {0:1000,1:5000})
t1 = time.time()
priced = proto.price_candidate(prog, {0:1000,1:5000})
json.dump(prog, open(f'tools/joint_search/out/{name}_table.json', 'w'))
print(f"{t1-t0:.4f} {priced}")
"""


def symbolic_flops_terms(cand_path):
    """truthmill's own symbolic_flops via harness flops --candidate."""
    # problem path is only used for sizes/domains; find it from the cand
    r = subprocess.run([HARNESS, "flops", cand_path[0],
                        "--candidate", cand_path[1]],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[:300])
    return json.loads(r.stdout)["symbolic_terms"]


def lead_flops_str(terms):
    """Highest-order symbolic term(s), truthmill order, display form."""
    deg = max(t["total_degree"] for t in terms)
    lead = [t for t in terms if t["total_degree"] == deg]
    parts = []
    for t in lead:
        c = t["coefficient"]
        names = {"0": "O", "1": "V"}
        mon = " ".join(f"{names[d]}^{e}" if e != 1 else names[d]
                       for d, e in sorted(t["powers"].items(),
                                          key=lambda kv: int(kv[0])))
        parts.append(f"{c}*{mon}" if c != "1" else mon)
    return " + ".join(parts)


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
        # flops + verification (symbolic polynomial via truthmill itself)
        ours_path = f"{ROOT}/tools/joint_search/out/{name}_table.json"
        g_cand = f"{COMP_DIR}/{name}_grist.json"
        ok_g = subprocess.run(
            [HARNESS, "check", prob, "--sizes", "0=3,1=5",
             "--candidate", g_cand], capture_output=True, text=True)
        passed_g = ok_g.returncode == 0 and json.loads(ok_g.stdout)["pass"]
        g_terms = symbolic_flops_terms((prob, g_cand))
        o_terms = symbolic_flops_terms(
            (prob, ours_path))
        ok = subprocess.run(
            [HARNESS, "check", prob, "--sizes", "0=3,1=5",
             "--candidate", ours_path],
            capture_output=True, text=True)
        passed = ok.returncode == 0 and json.loads(ok.stdout)["pass"]
        row = {"name": name,
               "grist_verified": passed_g,
               "grist_flops_symbolic_lead": lead_flops_str(g_terms),
               "ours_flops_symbolic_lead": lead_flops_str(o_terms),
               "grist_optimize_s": g_opt,
               "ours_solve_emit_s": float(o_solve),
               "verified": passed}
        row["grist_log10_flops"] = math.log10(
            proto.price_candidate(json.load(open(g_cand)), FLOP_SIZES))
        row["ours_log10_flops"] = math.log10(int(o_flops))
        rows.append(row)
        print(f"{name:12s} {row['grist_flops_symbolic_lead']:>24} | "
              f"{row['ours_flops_symbolic_lead']:>24} | "
              f"{g_opt:.2f}s | {o_solve}s | verified {passed} "
              f"(gristmill verified {passed_g})", flush=True)
    path = os.path.join(HERE, "trials",
                        time.strftime("%Y%m%d-%H%M%S") + ".jsonl")
    with open(path, "a") as fh:
        fh.write(json.dumps({"kind": "table-report", "load": load1(),
                             "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "rows": rows}) + "\n")
    print("logged to", path)


if __name__ == "__main__":
    main()
