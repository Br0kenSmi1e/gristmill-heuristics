#!/usr/bin/env python3
"""table_mpn.py — MPn ring-diagram comparison table, blog format.

Per order n (2..nmax): gristmill baseline vs the joint-search ILP —
  leading symbolic FLOP terms (truthmill symbolic_flops),
  optimizer runtime (gristmill optimize vs our solve+emit),
  verification (harness check at (3,5), both sides).
Our side runs under a 60 s budget + 3 GB RLIMIT_AS (the MPn sweep
found the factorial-canon cliff at n=6; timeouts are reported, not
hidden). Appends one JSON line to trials/.
"""
import json
import math
import os
import resource
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import proto  # noqa: E402
import gen_mpn  # noqa: E402
from table_report import (symbolic_flops_terms, lead_flops_str, load1,
                          HARNESS, GRIST_PY, GRIST_SCRIPT, COMP_DIR,
                          VENV_PY)

FLOP_SIZES = {0: 1000, 1: 5000}
BUDGET_S = 60

OURS_SOLVE = r"""
import sys, json, time
sys.path.insert(0, 'tools/joint_search')
import proto, gen_mpn
n = int(sys.argv[1])
prob = gen_mpn.mpn_problem(n)
proto.SYMS = prob['symmetries']
t0 = time.time()
pack = proto.solve_problem(prob, {0:1000,1:5000}, time_limit=45)
prog = proto.emit_global(pack, {0:1000,1:5000})
t1 = time.time()
priced = proto.price_candidate(prog, {0:1000,1:5000})
out = f'tools/joint_search/out/mpn{n}_table.json'
json.dump(prog, open(out, 'w'))
print(f"{t1-t0:.4f} {priced}")
"""


def _child_limits():
    resource.setrlimit(resource.RLIMIT_AS, (3 << 30, 3 << 30))


def run_row(n):
    prob_path = f"{COMP_DIR}/mpn{n}.json"
    with open(prob_path, "w") as fh:
        json.dump(gen_mpn.mpn_problem(n), fh)
    comp = f"{COMP_DIR}/mpn{n}_comp.json"
    with open(comp, "w") as fh:
        subprocess.run([HARNESS, "dump-comp", prob_path], stdout=fh,
                       check=True)

    row = {"name": f"mpn{n}"}

    # ---- gristmill baseline (its own 10-min guard; record slowness too)
    g_cand = f"{COMP_DIR}/mpn{n}_grist.json"
    try:
        r = subprocess.run([GRIST_PY, GRIST_SCRIPT, comp, g_cand],
                           capture_output=True, text=True, timeout=600)
        g_opt = None
        for line in r.stdout.splitlines():
            if line.startswith("TIMING"):
                parts = line.split()
                g_opt = float(parts[parts.index("optimize") + 1][:-1])
        if r.returncode != 0 or g_opt is None:
            row["grist"] = {"status": "crash", "stderr": r.stderr[-200:]}
        else:
            ok = subprocess.run([HARNESS, "check", prob_path, "--sizes",
                                 "0=3,1=5", "--candidate", g_cand],
                                capture_output=True, text=True)
            row["grist"] = {
                "flops_lead": lead_flops_str(
                    symbolic_flops_terms((prob_path, g_cand))),
                "log10_flops": math.log10(proto.price_candidate(
                    json.load(open(g_cand)), FLOP_SIZES)),
                "optimize_s": g_opt,
                "verified": ok.returncode == 0
                and json.loads(ok.stdout)["pass"]}
    except subprocess.TimeoutExpired:
        row["grist"] = {"status": "TIMEOUT", "budget_s": 600}

    # ---- ILP challenger (60 s budget)
    try:
        r2 = subprocess.run([VENV_PY, "-c", OURS_SOLVE, str(n)],
                            capture_output=True, text=True, cwd=ROOT,
                            timeout=BUDGET_S, preexec_fn=_child_limits)
        if r2.returncode != 0:
            row["ours"] = {"status": "OOM" if "MemoryError" in r2.stderr
                           else "crash", "stderr": r2.stderr[-200:]}
        else:
            o_solve, o_flops = r2.stdout.strip().split()[-2:]
            ours_path = f"{ROOT}/tools/joint_search/out/mpn{n}_table.json"
            ok = subprocess.run([HARNESS, "check", prob_path, "--sizes",
                                 "0=3,1=5", "--candidate", ours_path],
                                capture_output=True, text=True)
            row["ours"] = {
                "flops_lead": lead_flops_str(
                    symbolic_flops_terms((prob_path, ours_path))),
                "log10_flops": math.log10(int(o_flops)),
                "solve_emit_s": float(o_solve),
                "verified": ok.returncode == 0
                and json.loads(ok.stdout)["pass"]}
    except subprocess.TimeoutExpired:
        row["ours"] = {"status": "TIMEOUT", "budget_s": BUDGET_S}
    return row


def fmt(side, key, scale=1.0):
    if key == "flops_lead":
        return side.get(key, "—") if "status" not in side \
            else f"({side['status'].lower()})"
    if "status" in side:
        return "—"
    v = side.get(key)
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def main(nmax=8):
    if load1() > 8:
        sys.exit("load > 8, refusing (shared machine)")
    os.makedirs(COMP_DIR, exist_ok=True)
    rows = []
    for n in range(2, nmax + 1):
        row = run_row(n)
        rows.append(row)
        g, o = row["grist"], row["ours"]
        gv = {True: "✓", False: "✗"}.get(g.get("verified"), "—")
        ov = {True: "✓", False: "✗"}.get(o.get("verified"), "—")
        print(f"mpn{n:<4} {fmt(g,'flops_lead'):>26} | {fmt(o,'flops_lead'):>26} | "
              f"{fmt(g,'optimize_s'):>7} | {fmt(o,'solve_emit_s'):>7} | "
              f"{gv}/{ov}", flush=True)
    path = os.path.join(HERE, "trials",
                        time.strftime("%Y%m%d-%H%M%S") + "-mpn.jsonl")
    with open(path, "a") as fh:
        fh.write(json.dumps({"kind": "table-mpn", "load": load1(),
                             "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "rows": rows}) + "\n")
    print("logged to", path)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
