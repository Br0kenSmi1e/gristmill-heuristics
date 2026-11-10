#!/usr/bin/env python3
"""gen_cc.py — derive coupled-cluster residual equations via drudge (BCH)
and export them as truthmill problem JSON.

CC with T = T1+...+Tm: residuals R_mu = <Phi_mu| e^{-T} H e^T |0> for the
singles/doubles/.../m-tuples manifolds, with H the normal-ordered
particle-hole Hamiltonian (drudge's built-in PartHoleDrudge ham: Fock +
double-bar two-body). Quasiparticle convention: hole ops are flipped by
PartHoleDrudge's op parser, so excitation operators are written with
creation chars only (cr[a]..cr[i]).

Validation path: derive(m=2) is CCSD — compare against benches/chemistry/
ccsd.json (canonical signed term multisets per output). m=3 is CCSDT.

Runs with /tmp/gristmill-venv (drudge + dummy_spark shim; no JVM needed).
"""
import json
import os
import sys
from fractions import Fraction

sys.path.insert(0, "/home/longli/code/gristmill-heuristics/tools")
import dspark_shim  # noqa: F401  (patches collections for py3.12)

from dummy_spark import SparkContext                      # noqa: E402
from sympy import IndexedBase, Rational, symbols         # noqa: E402
from drudge import PartHoleDrudge                        # noqa: E402

# tensor id assignment (rank-4/6 antisym ids follow corpus conventions)
TID = {"f": 0, "u": 1, "t1": 2, "t2": 3, "t3": 4, "t4": 5, "t": 6}
SYMS = {
    "1": [{"permutation": [1, 0, 2, 3], "sign": -1},
          {"permutation": [0, 1, 3, 2], "sign": -1}],
    "3": [{"permutation": [1, 0, 2, 3], "sign": -1},
          {"permutation": [0, 1, 3, 2], "sign": -1}],
    "4": [{"permutation": [1, 0, 2, 3, 4, 5], "sign": -1},
          {"permutation": [0, 1, 2, 4, 3, 5], "sign": -1}],
    "5": [{"permutation": [1, 0, 2, 3, 4, 5, 6, 7], "sign": -1},
          {"permutation": [0, 1, 2, 3, 5, 6, 7, 4], "sign": -1}],
}


def make_drudge():
    dr = PartHoleDrudge(SparkContext())
    dr.full_simplify = False
    return dr


def cluster_ops(dr, m, factorial_norm=True):
    """T = sum_k T_k. With factorial_norm: T_k = (1/(k!)^2) sum t_k...
    (standard CC convention). Without: plain sum t_k[...] ops (the MP
    recursion convention — matches corpus mp3 coefficients)."""
    import math
    p = dr.names
    T = None
    bases = {}
    for k in range(1, m + 1):
        t = IndexedBase(f"t{k}")
        dr.set_dbbar_base(t, k)
        bases[k] = t
        aa = p.V_dumms[:k]
        ii = p.O_dumms[:k]
        ops = dr.cr[aa[0]]
        for x in aa[1:]:
            ops = ops * dr.cr[x]
        for x in ii:
            ops = ops * dr.an[x]
        sums = [(x, p.V) for x in aa] + [(x, p.O) for x in ii]
        c = Rational(1, math.factorial(k) ** 2) if factorial_norm \
            else Rational(1)
        Tk = dr.sum(*sums, c * t[aa + ii] * ops)
        T = Tk if T is None else T + Tk
    return T.simplify(), bases


def comm(A, B):
    return A * B - B * A


def hbar(dr, T, order=4):
    """Nested-commutator BCH: e^{-T} H e^T to the given order (2-body H
    terminates exactly at order 4)."""
    H = dr.ham + dr.ham_energy
    res = H
    term = H
    for n in range(1, order + 1):
        term = comm(term, T).simplify() / n
        res = res + term
        print(f"  BCH order {n}: {term.n_terms} terms", flush=True)
    return res.simplify()


def project(hb, level):
    """Extract the level-k residual: Hbar terms whose field operators are
    exactly k particle creations + k hole creations (quasiparticle
    convention; level 0 = energy). The vec indices (dummies over the
    respective ranges) become the free output labels; the amplitude is
    untouched. Returns (terms, n_free_holes, n_free_parts)."""
    out = []
    for term in hb.terms.collect():
        vecs = term.vecs
        if len(vecs) != 2 * level:
            continue
        sums = dict(term.sums)
        rngs = sorted((str(v.indices[0]), sums[v.indices[1]].label)
                      for v in vecs)
        if rngs != [("AN", "O")] * level + [("CR", "V")] * level:
            continue
        out.append(term)
    return out


def export_residual(terms, level, values):
    """Export one residual sector as an output: one Einsum value per term
    + a LinComb. Free labels = the vec indices, ordered holes first
    (positions 0..level-1) then particles, matching the corpus output
    convention (i,j,...,a,b,...)."""
    r = 2 * level
    term_ids = []
    for term in terms:
        coeff, factors = term.amp.as_coeff_mul()
        coeff = Rational(coeff)
        sums = dict(term.sums)
        vec_idx = [v.indices[1] for v in term.vecs]
        holes = [s for s in vec_idx if sums[s].label == "O"]
        parts = [s for s in vec_idx if sums[s].label == "V"]
        label_of = {}
        dom = {}
        for q, s in enumerate(holes + parts):
            label_of[s] = q
            dom[q] = 0 if q < level else 1

        def lab(s):
            if s not in label_of:
                label_of[s] = len(label_of)
                rng = sums[s]
                dom[label_of[s]] = 0 if rng.label == "O" else 1
            return label_of[s]

        inputs, args = [], []
        flat = []
        for f in factors:
            if f.is_Pow and f.exp.is_integer and f.exp > 0:
                flat.extend([f.base] * int(f.exp))
            else:
                flat.append(f)
        for f in flat:
            assert hasattr(f, "indices"), f"unexpected factor {f}"
            name = str(f.base)
            args.append({"Tensor": TID[name]})
            inputs.append([lab(ix) for ix in f.indices])
        values.append({"Einsum": {
            "coeff": [coeff.numerator, coeff.denominator],
            "code": {"inputs": inputs, "output": list(range(r)),
                     "domains": {str(k): v for k, v in dom.items()}},
            "args": args}})
        term_ids.append(len(values) - 1)
    if not term_ids:
        return None
    if len(term_ids) == 1:
        return term_ids[0]
    free_dom = {str(q): (0 if q < level else 1) for q in range(r)}
    values.append({"LinComb": {
        "coeff": [[1, 1]] * len(term_ids),
        "code": {"inputs": [list(range(r))] * len(term_ids),
                 "output": list(range(r)), "domains": free_dom},
        "args": [{"Value": i} for i in term_ids]}})
    return len(values) - 1


def derive(m, verbose=True):
    dr = make_drudge()
    if verbose:
        print(f"deriving CC with T1..T{m} ...", flush=True)
    T, _bases = cluster_ops(dr, m)
    if verbose:
        print("T built:", T.n_terms, "terms; BCH ...", flush=True)
    hb = hbar(dr, T)
    if verbose:
        print("Hbar:", hb.n_terms, "terms; projecting ...", flush=True)
    values, outputs = [], []
    for level in range(0, m + 1):
        terms = project(hb, level)
        if verbose:
            print(f"  residual level {level}: {len(terms)} terms",
                  flush=True)
        ov = export_residual(terms, level, values)
        if ov is not None:
            outputs.append(ov)
    # declare every tensor id used (symmetries map doubles as declaration)
    used = set()
    for v in values:
        code = v.get("Einsum", {}).get("code") if "Einsum" in v else None
        if code is None:
            continue
        for a in v["Einsum"]["args"]:
            if "Tensor" in a:
                used.add(str(a["Tensor"]))
    prob = {"sizes": {"0": 3, "1": 5},
            "symmetries": {k: SYMS.get(k, []) for k in sorted(used)},
            "reference": {"values": values, "outputs": outputs}}
    return prob


if __name__ == "__main__":
    m = int(sys.argv[1])
    out = sys.argv[2]
    prob = derive(m)
    with open(out, "w") as f:
        json.dump(prob, f)
    print(f"wrote {out}: {len(prob['reference']['outputs'])} outputs, "
          f"{len(prob['reference']['values'])} values")
