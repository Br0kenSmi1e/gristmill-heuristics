#!/usr/bin/env python3
"""MPn-style ring-diagram problem generator + scaling sweep.

Ring_n = scalar contraction of n rank-4 tensors in a cycle:

    prod_k  X_k[h_k, h_{k+1}, p_k, p_{k+1}]     (k mod n; h in o, p in v)

with X antisymmetric in (0,1) and (2,3) — the chemistry-corpus
convention for the two-electron integral / T2 vertex. Each order-n
problem has THREE scalar outputs (pure-V ring, T.V^{n-1}, T2.V^{n-2}
rings), which share massive substructure, so cross-output sharing and
the full family/cluster machinery are exercised.

Scaling knobs per order n: factors/term = n (subset axis 2^n) and
intermediate rank ~ 2(m+1) for m-factor chains (canon axis, (m+1)!^2
renaming groups). MP2/MP3 ring diagrams are n=2,3.

Subcommands (run from tools/joint_search):
  gen <n> <out.json>    write the order-n problem JSON
  solve1 <n>            solve one instance, print one stats JSON line
  sweep [nmax]          sweep n=2..nmax, 60 s budget per instance,
                        append lines to trials/mpn_sweep_<ts>.jsonl
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

V, T = 1, 3        # tensor ids, matching the ccsd corpus convention
GENS = [{"permutation": [1, 0, 2, 3], "sign": -1},
        {"permutation": [0, 1, 3, 2], "sign": -1}]


def ring_value(tids, n):
    inputs = [[k, (k + 1) % n, n + k, n + (k + 1) % n] for k in range(n)]
    dom = {str(l): (0 if l < n else 1) for l in range(2 * n)}
    return {"Einsum": {"coeff": [1, 1],
                       "code": {"inputs": inputs, "output": [],
                                "domains": dom},
                       "args": [{"Tensor": t} for t in tids]}}


def mpn_problem(n):
    values = [ring_value([V] * n, n),
              ring_value([T] + [V] * (n - 1), n),
              ring_value([T, T] + [V] * (n - 2), n)]
    return {"sizes": {"0": 3, "1": 5},
            "symmetries": {str(V): GENS, str(T): GENS},
            "reference": {"values": values, "outputs": [0, 1, 2]}}


def cmd_gen(n, out):
    with open(out, "w") as f:
        json.dump(mpn_problem(n), f)
    print(f"wrote {out}: order-{n} rings, 3 outputs, {n} factors/term")


def cmd_solve1(n):
    import proto
    prob = mpn_problem(n)
    proto.SYMS = prob["symmetries"]
    calls = [0]
    orig = proto.canon_full
    proto.canon_full = lambda *a, **k: (calls.__setitem__(0, calls[0] + 1),
                                        orig(*a, **k))[1]
    t0 = time.time()
    res = proto.solve_problem(prob, {0: 16, 1: 64}, time_limit=45)
    total = time.time() - t0
    if res is None:
        print(json.dumps({"n": n, "status": "no-solution",
                          "total_s": round(total, 2)}))
        return
    st = res["stats"]
    print(json.dumps({
        "n": n, "U": st["U"], "cand": st["candidates"],
        "vars": st["vars"], "cons": st["constraints"],
        "fams": st["families"], "canon_calls": calls[0],
        "build_s": round(st["build_s"], 3), "total_s": round(total, 2),
        "ilp_status": st["status"], "mip_gap": st["mip_gap"],
        "flops": st["flops_at_solution"]}))


def _child_limits():
    """preexec guard: never let one instance pressure the box (14 GB)."""
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (3 << 30, 3 << 30))


def cmd_sweep(nmax, budget=60):
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(HERE, "trials", f"mpn_sweep_{ts}.jsonl")
    for n in range(2, nmax + 1):
        try:
            r = subprocess.run([sys.executable, os.path.abspath(__file__),
                                "solve1", str(n)],
                               capture_output=True, text=True,
                               timeout=budget, preexec_fn=_child_limits)
            if r.returncode != 0:
                oom = "MemoryError" in r.stderr
                line = json.dumps({"n": n,
                                   "status": "OOM" if oom else "crash",
                                   "stderr": r.stderr[-200:]})
            else:
                line = r.stdout.strip().splitlines()[-1] \
                    if r.stdout.strip() else json.dumps(
                        {"n": n, "status": "no-output"})
        except subprocess.TimeoutExpired:
            line = json.dumps({"n": n, "status": "TIMEOUT",
                               "budget_s": budget})
        with open(out, "a") as f:
            f.write(line + "\n")
        print(line, flush=True)
    print(f"-> {out}")


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        cmd_gen(int(sys.argv[2]), sys.argv[3])
    elif sys.argv[1] == "solve1":
        cmd_solve1(int(sys.argv[2]))
    elif sys.argv[1] == "sweep":
        cmd_sweep(int(sys.argv[2]) if len(sys.argv) > 2 else 11)
