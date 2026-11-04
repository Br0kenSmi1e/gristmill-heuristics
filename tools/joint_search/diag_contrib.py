#!/usr/bin/env python3
"""Numeric per-contribution bisector for B-006.

Evaluates the emitted program and the reference terms in numpy at the
fixture sizes with symmetry-satisfying random tensors, then reports, for
each final-LinComb contribution, which true term(s) it matches and the
residual. Usage:

  diag_contrib.py <fixture.json> <candidate.json>
"""
import json
import sys
from fractions import Fraction

import numpy as np


def _patterns_of(program):
    """Ordered (tid, pattern) collection, matching harness collect_patterns."""
    pats = {}
    for v in program["values"]:
        if "Einsum" not in v:
            continue
        d = v["Einsum"]
        for arg, inp in zip(d["args"], d["code"]["inputs"]):
            if "Tensor" in arg:
                pat = tuple(d["code"]["domains"][str(l)] for l in inp)
                pats.setdefault(arg["Tensor"], [])
                if pat not in pats[arg["Tensor"]]:
                    pats[arg["Tensor"]].append(pat)
    return pats


def make_tensors(prob, sizes, seed=0, extra_programs=()):
    """Replicate harness dense::generate_inputs_for: one block per
    (tensor, domain-pattern); patterns related by a generator share a
    representative (projected on its stabilizer); propagation by
    permute+sign along the FULL pattern orbit."""
    pats = _patterns_of(prob["reference"])
    for prog in extra_programs:
        for tid, ps in _patterns_of(prog).items():
            pats.setdefault(tid, [])
            for p in ps:
                if p not in pats[tid]:
                    pats[tid].append(p)
    rng = np.random.default_rng(seed)
    blocks = {}
    for tid in sorted(pats):
        gens = prob["symmetries"].get(str(tid), [])
        filled = set()
        for rep in pats[tid]:
            if rep in filled:
                continue
            dims = tuple(sizes[str(d)] for d in rep)
            block = rng.standard_normal(dims)
            stab = _stabilizer(gens, rep)
            if len(stab) > 1:
                acc = np.zeros_like(block)
                for perm, s in stab:
                    acc = acc + s * np.transpose(block, perm)
                block = acc / len(stab)
            queue = [(rep, block)]
            while queue:
                cur_key, cur = queue.pop()
                if (tid, cur_key) in blocks:
                    continue
                blocks[(tid, cur_key)] = cur
                filled.add(cur_key)
                for g in gens:
                    p = g["permutation"]
                    q = tuple(cur_key[p[k]] for k in range(len(p)))
                    if q in filled:
                        continue
                    image = g["sign"] * np.transpose(cur, p)
                    queue.append((q, image))
    return blocks


def _stabilizer(gens, pattern):
    """Closure of pattern-preserving generators, as (perm, sign) list."""
    rank = len(pattern)
    group = [(tuple(range(rank)), 1)]
    queue = [(tuple(g["permutation"]), g["sign"]) for g in gens
             if all(pattern[p] == pattern[k]
                    for k, p in enumerate(g["permutation"]))]
    while queue:
        perm, sign = queue.pop()
        cands = []
        for gp, gs in group:
            cands.append((tuple(perm[gp[k]] for k in range(rank)),
                          sign * gs))
            cands.append((tuple(gp[perm[k]] for k in range(rank)),
                          sign * gs))
        for c in cands:
            if not any(p == c[0] for p, _ in group):
                group.append(c)
                queue.append(c)
    return group


def eval_einsum(d, vals, blocks):
    args = []
    for a, inp in zip(d["args"], d["code"]["inputs"]):
        if "Tensor" in a:
            pat = tuple(d["code"]["domains"][str(l)] for l in inp)
            args.append(blocks[(a["Tensor"], pat)])
        else:
            args.append(vals[a["Value"]])
    labels = sorted({l for inp in d["code"]["inputs"] for l in inp}
                    | set(d["code"]["output"]))
    letters = {l: chr(ord("a") + i) for i, l in enumerate(labels)}
    eq = ",".join("".join(letters[l] for l in inp)
                  for inp in d["code"]["inputs"]) \
        + "->" + "".join(letters[l] for l in d["code"]["output"])
    c = Fraction(*d["coeff"]) if isinstance(d["coeff"][0], int) else 1
    return float(c) * np.einsum(eq, *args)


def eval_program(prog, blocks, ntensors=None):
    vals = {}
    for i, v in enumerate(prog["values"]):
        if "Einsum" in v:
            vals[i] = eval_einsum(v["Einsum"], vals, blocks)
        else:
            d = v["LinComb"]
            acc = None
            for cf, a, inp in zip(d["coeff"], d["args"],
                                  d["code"]["inputs"]):
                if "Tensor" in a:
                    pat = tuple(d["code"]["domains"][str(l)] for l in inp)
                    t = blocks[(a["Tensor"], pat)]
                else:
                    t = vals[a["Value"]]
                # input labels map positionally onto the arg's axes;
                # reorder to the LinComb's output label order
                out = d["code"]["output"]
                if list(inp) != list(out):
                    perm = [inp.index(l) for l in out]
                    t = np.transpose(t, perm)
                term = float(Fraction(*cf)) * t
                acc = term if acc is None else acc + term
            vals[i] = acc
    return vals


def main():
    prob = json.load(open(sys.argv[1]))
    prog = json.load(open(sys.argv[2]))
    sizes = prob["sizes"]
    blocks = make_tensors(prob, sizes, extra_programs=[prog])

    # true terms
    terms = []
    for v in prob["reference"]["values"]:
        if "Einsum" in v:
            terms.append(eval_einsum(v["Einsum"], {}, blocks))
    true_total = sum(terms)

    vals = eval_program(prog, blocks)
    out = prog["outputs"][0]
    got = vals[out]
    err = np.max(np.abs(got - true_total)) / max(1e-30, np.max(np.abs(true_total)))
    print(f"total rel err: {err:.3e}")

    # per-contribution of the final lincomb
    fin = prog["values"][out]["LinComb"]
    covered = set()
    for ci, (cf, a) in enumerate(zip(fin["coeff"], fin["args"])):
        contrib = float(Fraction(*cf)) * vals[a["Value"]]
        # match against subsets: single term or perm-pair of terms
        best = None
        for ti, t in enumerate(terms):
            e = np.max(np.abs(contrib - t)) / max(1e-30, np.max(np.abs(t)))
            if best is None or e < best[0]:
                best = (e, ti)
        print(f"contrib {ci} (Value {a['Value']}, coeff {cf}): "
              f"best single-term match {best[1]} rel {best[0]:.3e}")
        if best[0] < 1e-9:
            covered.add(best[1])
    print("covered terms:", sorted(covered), "of", len(terms))
    # residual: does sum of contribs == total?
    missing = [t for i, t in enumerate(terms) if i not in covered]
    if missing:
        m = sum(missing)
        print(f"missing-term mass rel: "
              f"{np.max(np.abs(m))/max(1e-30,np.max(np.abs(true_total))):.3e}")


if __name__ == "__main__":
    main()
