#!/usr/bin/env python3
"""gen_mp.py — derive REAL Moller-Plesset MPn energy expressions by
perturbative expansion of the coupled-cluster equations, and export them
as truthmill problem JSON.

Method: the CC residual equations (from gen_cc's drudge pipeline, m=4 =
CCSDTQ, canonical HF so the Fock matrix is diagonal and its terms are
dropped) are polynomials in the amplitudes. Expanding T_s = sum_k
lambda^k t_s^(k) and collecting orders gives the recursion

    t_s^(k) = d_s  .  (order-k terms of residual_s with lower-order
                       amplitudes substituted)

where d_s is the denominator tensor 1/(sum eps_holes - sum eps_parts),
symmetric (+1) under within-half swaps (corpus mp3 convention: d is
tensor 1, v is tensor 0, particle-first index order). Energy:
E_n = (1/4) v . t2^(n-1). Fully expanded, the MPn energy is a polynomial
in v and d_* only — matching the corpus mp3 form (3 v + 2 d, 6 terms).

Validation: mp3 derivation must reproduce corpus mp3 sig-for-sig.

Runs with /tmp/gristmill-venv (drudge + dummy_spark).
"""
import json
import sys
from fractions import Fraction
from itertools import product as iproduct

sys.path.insert(0, "/home/longli/code/gristmill-heuristics/tools")
import dspark_shim  # noqa: F401

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gen_cc import make_drudge, cluster_ops, hbar, project  # noqa: E402

AMP_SECTOR = {"t1": 1, "t2": 2, "t3": 3, "t4": 4}

# tensor ids, corpus mp3 conventions
TID = {"u": 0, "d2": 1, "d3": 7, "d4": 8, "d1": 9}
SYMS = {
    # v carries the FULL physical symmetry: within-half antisymmetry plus
    # half-exchange v_pqrs = v_rspq (corpus declares only the antisym
    # pair — a latent sharing opportunity there)
    "0": [{"permutation": [1, 0, 2, 3], "sign": -1},
          {"permutation": [0, 1, 3, 2], "sign": -1},
          {"permutation": [2, 3, 0, 1], "sign": 1}],
    "1": [{"permutation": [1, 0, 2, 3], "sign": 1},
          {"permutation": [0, 1, 3, 2], "sign": 1}],
    "7": [{"permutation": [1, 0, 2, 3, 4, 5], "sign": 1},
          {"permutation": [1, 2, 0, 3, 4, 5], "sign": 1},
          {"permutation": [0, 1, 2, 4, 3, 5], "sign": 1},
          {"permutation": [0, 1, 2, 4, 5, 3], "sign": 1}],
    "8": [{"permutation": [1, 0, 2, 3, 4, 5, 6, 7], "sign": 1},
          {"permutation": [1, 2, 3, 0, 4, 5, 6, 7], "sign": 1},
          {"permutation": [0, 1, 2, 3, 5, 4, 6, 7], "sign": 1},
          {"permutation": [0, 1, 2, 3, 5, 6, 7, 4], "sign": 1}],
}

_fresh = [0]


def fresh():
    from sympy import Symbol
    _fresh[0] += 1
    return Symbol(f"zz{_fresh[0]}")


def sector_ir(hb, level):
    """Residual sector as neutral IR terms:
    {coeff: Fraction, factors: [(base, idxs)], sums: {sym: 'O'|'V'},
     free: symbols in t-order (particles..., holes...)}.
    Terms containing the Fock base f are dropped (canonical HF)."""
    out = []
    for term in project(hb, level):
        coeff, factors = term.amp.as_coeff_mul()
        sums = {s: r.label for s, r in term.sums}
        parts = [v.indices[1] for v in term.vecs
                 if str(v.indices[0]) == "CR"]
        holes = [v.indices[1] for v in term.vecs
                 if str(v.indices[0]) == "AN"]
        flat = []
        for f in factors:
            if f.is_Pow and f.exp.is_integer and f.exp > 0:
                flat.extend([f.base] * int(f.exp))
            else:
                flat.append(f)
        facs = [(str(f.base), tuple(f.indices)) for f in flat]
        if any(b == "f" for b, _ in facs):
            continue
        out.append({"coeff": Fraction(coeff.numerator, coeff.denominator),
                    "factors": facs, "sums": sums,
                    "free": tuple(parts + holes)})
    return out


def splice(term, j, mu, idxs, sums_mu):
    """term with factor j replaced by IR term mu, whose free symbols map
    to the occurrence's idxs; mu's dummies alpha-renamed fresh."""
    ren = {}
    for s in sums_mu:
        ren[s] = fresh()
    for q, s in enumerate(mu["free"]):
        ren[s] = idxs[q]
    factors = (term["factors"][:j]
               + [(b, tuple(ren[x] for x in ix)) for b, ix in mu["factors"]]
               + term["factors"][j + 1:])
    sums = dict(term["sums"])
    for s, r in sums_mu.items():
        sums[ren[s]] = r
    return {"coeff": term["coeff"] * mu["coeff"], "factors": factors,
            "sums": sums, "free": term["free"]}


def compositions(total, m, lo=1):
    if m == 0:
        if total == 0:
            yield ()
        return
    for first in range(lo, total - lo * (m - 1) + 1):
        for rest in compositions(total - first, m - 1, lo):
            yield (first,) + rest


class MP:
    def __init__(self, residuals):
        self.res = residuals      # sector -> [IR terms]
        self.memo = {}            # (s, k) -> expanded IR terms (u/d only)

    def amp(self, s, k):
        """t_s^(k) fully expanded (bases u, d_* only), free in t-order."""
        if (s, k) in self.memo:
            return self.memo[(s, k)]
        base = f"t{s}"
        out = []
        if k >= 1:
            for t in self.res[s]:
                pos = [j for j, (b, _) in enumerate(t["factors"])
                       if b in AMP_SECTOR]
                for ks in compositions(k - 1, len(pos)):
                    # substitute LAST amp factor first: splicing changes
                    # the factor list length, stale positions otherwise
                    cur = [t]
                    for j, kj in sorted(zip(pos, ks), reverse=True):
                        sj = AMP_SECTOR[t["factors"][j][0]]
                        nxt = []
                        for tt in cur:
                            for mu in self.amp(sj, kj):
                                nxt.append(splice(tt, j, mu,
                                                  tt["factors"][j][1],
                                                  mu["sums"]))
                        cur = nxt
                    out.extend(cur)
        # multiply by d_s (indices = free labels in t-order)
        res = [{"coeff": t["coeff"],
                "factors": [(f"d{s}", t["free"])] + t["factors"],
                "sums": t["sums"], "free": t["free"]} for t in out]
        self.memo[(s, k)] = res
        return res

    def energy(self, n):
        """E_n: order-n substituted energy-sector terms (u/d only)."""
        out = []
        for t in self.res[0]:
            pos = [j for j, (b, _) in enumerate(t["factors"])
                   if b in AMP_SECTOR]
            for ks in compositions(n - 1, len(pos)):
                cur = [t]
                for j, kj in sorted(zip(pos, ks), reverse=True):
                    sj = AMP_SECTOR[t["factors"][j][0]]
                    nxt = []
                    for tt in cur:
                        for mu in self.amp(sj, kj):
                            nxt.append(splice(tt, j, mu,
                                              tt["factors"][j][1],
                                              mu["sums"]))
                    cur = nxt
                out.extend(cur)
        return out


def emit_problem(terms):
    """IR terms (scalar, u/d_* bases) -> truthmill problem JSON."""
    values = []
    for t in terms:
        label_of = {}
        dom = {}

        def lab(s, t=t, label_of=label_of, dom=dom):
            if s not in label_of:
                label_of[s] = len(label_of)
                dom[label_of[s]] = 0 if t["sums"][s] == "O" else 1
            return label_of[s]

        inputs = [[lab(x) for x in ix] for _b, ix in t["factors"]]
        values.append({"Einsum": {
            "coeff": [t["coeff"].numerator, t["coeff"].denominator],
            "code": {"inputs": inputs, "output": [],
                     "domains": {str(k): v for k, v in dom.items()}},
            "args": [{"Tensor": TID[b]} for b, _ in t["factors"]]}})
    used = sorted({str(TID[b]) for t in terms for b, _ in t["factors"]})
    if len(values) == 1:
        outputs = [0]
    else:
        values.append({"LinComb": {
            "coeff": [[1, 1]] * len(values),
            "code": {"inputs": [[]] * (len(values)),
                     "output": [], "domains": {}},
            "args": [{"Value": i} for i in range(len(values))]}})
        outputs = [len(values) - 1]
    return {"sizes": {"0": 3, "1": 5},
            "symmetries": {k: SYMS.get(k, []) for k in used},
            "reference": {"values": values, "outputs": outputs}}


def derive_residuals(m=4):
    dr = make_drudge()
    T, _ = cluster_ops(dr, m, factorial_norm=False)
    hb = hbar(dr, T)
    residuals = {}
    for level in range(0, m + 1):
        residuals[level] = sector_ir(hb, level)
        print(f"  sector {level}: {len(residuals[level])} terms (f dropped)",
              flush=True)
    return residuals


if __name__ == "__main__":
    n = int(sys.argv[1])
    out = sys.argv[2]
    res = derive_residuals(4)
    mp = MP(res)
    en = mp.energy(n)
    print(f"MP{n} energy: {len(en)} terms, maxfac "
          f"{max(len(t['factors']) for t in en)}")
    prob = emit_problem(en)
    with open(out, "w") as f:
        json.dump(prob, f)
    print(f"wrote {out}")
