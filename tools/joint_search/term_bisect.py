#!/usr/bin/env python3
"""Term-subset bisector (ddmin) for verify failures.

Extracts subsets of one output definition's terms as standalone problems
(same convention as make_fixtures.py), emits with sym, and harness-checks
at (3,5). Minimizes a failing subset.

Usage (from worktree root):
  tools/joint_search/.venv/bin/python tools/joint_search/bisect.py \
      benches/chemistry/ccsd.json 82 [start subset as comma list]
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import proto  # noqa: E402

HARNESS = os.path.join(ROOT, "target", "debug", "harness")
SIZES = {0: 3, 1: 5}


def make_subproblem(problem, ov, idxs):
    proto.SYMS = problem["symmetries"]
    terms_all, r = proto.parse_terms(problem, ov)
    values = []
    for i in idxs:
        t = terms_all[i]
        values.append({"Einsum": {
            "coeff": [t["coeff"].numerator, t["coeff"].denominator],
            "code": {"inputs": [list(ls) for _, ls in t["factors"]],
                     "output": list(range(r)),
                     "domains": {str(l): d for l, d in t["dom"].items()}},
            "args": [{"Tensor": tid} for tid, _ in t["factors"]]}})
    out_doms = {str(l): terms_all[idxs[0]]["dom"][l] for l in range(r)}
    values.append({"LinComb": {"coeff": [[1, 1]] * len(idxs),
        "code": {"inputs": [list(range(r))] * len(idxs),
                 "output": list(range(r)), "domains": out_doms},
        "args": [{"Value": i} for i in range(len(idxs))]}})
    return {"sizes": {"0": 3, "1": 5}, "symmetries": problem["symmetries"],
            "reference": {"values": values, "outputs": [len(values) - 1]}}


def fails(problem, ov, idxs, workdir, tag):
    """True if the emitted sym candidate fails verify/oracle."""
    sub = make_subproblem(problem, ov, idxs)
    spath = os.path.join(workdir, f"sub_{tag}.json")
    cpath = os.path.join(workdir, f"cand_{tag}.json")
    json.dump(sub, open(spath, "w"))
    proto.SYMS = sub["symmetries"]
    pack = proto.solve_problem(sub, SIZES)
    prog = proto.emit_global(pack, SIZES)
    json.dump(prog, open(cpath, "w"))
    r = subprocess.run([HARNESS, "check", spath, "--sizes", "0=3,1=5",
                        "--candidate", cpath],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return True
    return not json.loads(r.stdout)["pass"]


def ddmin(problem, ov, idxs, workdir):
    """Classic delta debugging on the term-index list."""
    idxs = list(idxs)
    n = 2
    tests = 0
    while len(idxs) >= 2:
        chunk = (len(idxs) + n - 1) // n
        reduced = False
        for i in range(0, len(idxs), chunk):
            comp = idxs[:i] + idxs[i + chunk:]
            if not comp:
                continue
            tests += 1
            if fails(problem, ov, comp, workdir, f"dd{tests}"):
                print(f"  drop {len(idxs)-len(comp)} -> {sorted(comp)}",
                      flush=True)
                idxs = comp
                n = max(n - 1, 2)
                reduced = True
                break
        if reduced:
            continue
        if n >= len(idxs):
            break
        n = min(len(idxs), n * 2)
    return sorted(idxs), tests


def main():
    prob_path, ov = sys.argv[1], int(sys.argv[2])
    problem = json.load(open(prob_path))
    proto.SYMS = problem["symmetries"]
    terms_all, r = proto.parse_terms(problem, ov)
    if len(sys.argv) > 3:
        start = [int(x) for x in sys.argv[3].split(",")]
    else:
        start = list(range(len(terms_all)))
    workdir = tempfile.mkdtemp(prefix="bisect_")
    print(f"{len(terms_all)} terms, start subset {len(start)}, "
          f"workdir {workdir}", flush=True)
    assert fails(problem, ov, start, workdir, "start"), \
        "start subset does not fail"
    minimal, tests = ddmin(problem, ov, start, workdir)
    print(f"minimal failing subset ({len(minimal)} terms): {minimal}")
    # verify minimality: every proper single-deletion passes
    for i in minimal:
        comp = [x for x in minimal if x != i]
        if comp and fails(problem, ov, comp, workdir, "checkmin"):
            print(f"  WARNING: not minimal, {comp} also fails")


if __name__ == "__main__":
    main()
