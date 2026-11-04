#!/usr/bin/env python3
"""Regenerate the synthetic fixtures under fixtures/. Deterministic.
Run from the worktree root: python3 tools/joint_search/make_fixtures.py"""

import json
import os

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "fixtures")
os.makedirs(OUT, exist_ok=True)

def dump(name, prob):
    json.dump(prob, open(os.path.join(OUT, name), "w"))
    print("wrote", name)

# antisym_test: Z[i,j] = sum_ab V[a,b]A[a,i]B[b,j] + V[b,a]A[b,i]B[a,j]
# with V antisymmetric: term2 = -perm(term1)  =>  Z = M - M^T
dump("antisym_test.json", {
    "sizes": {"0": 3},
    "symmetries": {"0": [{"permutation": [1, 0], "sign": -1}],
                   "1": [], "2": []},
    "reference": {"values": [
        {"Einsum": {"coeff": [1, 1],
            "code": {"inputs": [[0,1],[0,2],[1,3]], "output": [2,3],
                     "domains": {"0": 0, "1": 0, "2": 0, "3": 0}},
            "args": [{"Tensor": 0}, {"Tensor": 1}, {"Tensor": 2}]}},
        {"Einsum": {"coeff": [1, 1],
            "code": {"inputs": [[1,0],[1,2],[0,3]], "output": [2,3],
                     "domains": {"0": 0, "1": 0, "2": 0, "3": 0}},
            "args": [{"Tensor": 0}, {"Tensor": 1}, {"Tensor": 2}]}},
        {"LinComb": {"coeff": [[1,1],[1,1]],
            "code": {"inputs": [[2,3],[2,3]], "output": [2,3],
                     "domains": {"2": 0, "3": 0}},
            "args": [{"Value": 0}, {"Value": 1}]}}],
        "outputs": [2]},
})

# antisym_test2: perm pair + shared subterm + one more pair
def einsum(coeff, inputs, output, args):
    doms = {}
    for inp in inputs + [output]:
        for l in inp:
            doms[str(l)] = 0
    return {"Einsum": {"coeff": coeff,
        "code": {"inputs": inputs, "output": output, "domains": doms},
        "args": [{"Tensor": a} for a in args]}}

dump("antisym_test2.json", {
    "sizes": {"0": 3},
    "symmetries": {"0": [{"permutation": [1, 0], "sign": -1}],
                   "1": [], "2": [], "3": []},
    "reference": {"values": [
        einsum([1,1], [[0,1],[0,4],[1,5]], [4,5], [0,1,2]),
        einsum([1,1], [[1,0],[1,4],[0,5]], [4,5], [0,1,2]),
        einsum([1,1], [[0,1],[0,5],[1,4]], [4,5], [0,1,2]),
        einsum([1,1], [[1,0],[1,5],[0,4]], [4,5], [0,1,2]),
        einsum([1,1], [[0,1],[0,4],[1,6],[6,5]], [4,5], [0,1,2,3]),
        {"LinComb": {"coeff": [[1,1]]*5,
            "code": {"inputs": [[4,5]]*5, "output": [4,5],
                     "domains": {"4": 0, "5": 0}},
            "args": [{"Value": i} for i in range(5)]}}],
        "outputs": [5]},
})

# ccsd_energy: the 3-term scalar CCSD energy definition
dump("ccsd_energy.json", {
    "sizes": {"0": 3, "1": 5},
    "symmetries": {"0": [], "2": [],
                   "1": [{"permutation": [1,0,2,3], "sign": -1},
                         {"permutation": [0,1,3,2], "sign": -1}],
                   "3": [{"permutation": [1,0,2,3], "sign": -1},
                         {"permutation": [0,1,3,2], "sign": -1}]},
    "reference": {"values": [
        {"Einsum": {"coeff": [1,1],
            "code": {"inputs": [[0,1],[1,0]], "output": [],
                     "domains": {"0": 0, "1": 1}},
            "args": [{"Tensor": 0}, {"Tensor": 2}]}},
        {"Einsum": {"coeff": [1,4],
            "code": {"inputs": [[0,1,2,3],[2,3,0,1]], "output": [],
                     "domains": {"0":1,"1":1,"2":0,"3":0}},
            "args": [{"Tensor": 3}, {"Tensor": 1}]}},
        {"Einsum": {"coeff": [1,2],
            "code": {"inputs": [[0,1],[2,3],[1,3,0,2]], "output": [],
                     "domains": {"0":1,"1":0,"2":1,"3":0}},
            "args": [{"Tensor": 2}, {"Tensor": 2}, {"Tensor": 1}]}},
        {"LinComb": {"coeff": [[1,1]]*3,
            "code": {"inputs": [[],[],[]], "output": [], "domains": {}},
            "args": [{"Value": 0}, {"Value": 1}, {"Value": 2}]}}],
        "outputs": [3]},
})

# ccsd_doubles minimal failing set for B-006: terms {32,33,37,42},
# extracted from benches/chemistry/ccsd.json via proto.parse_terms.
import sys
sys.path.insert(0, HERE)
import proto  # noqa: E402

problem = json.load(open("benches/chemistry/ccsd.json"))
proto.SYMS = problem["symmetries"]
terms_all, r = proto.parse_terms(problem, 82)
idxs = [38, 39, 40, 42, 43]
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
values.append({"LinComb": {"coeff": [[1,1]]*len(idxs),
    "code": {"inputs": [list(range(r))]*len(idxs),
             "output": list(range(r)), "domains": out_doms},
    "args": [{"Value": i} for i in range(len(idxs))]}})
dump("ccsd_doubles_38_39_40_42_43.json",
     {"sizes": {"0": 3, "1": 5}, "symmetries": problem["symmetries"],
      "reference": {"values": values, "outputs": [len(values)-1]}})
