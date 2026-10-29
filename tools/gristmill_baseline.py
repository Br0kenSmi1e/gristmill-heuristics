#!/usr/bin/env python3
"""Build a gristmill baseline for a truthmill problem: take the EXACT same
equations (expanded rewrites-Computation JSON from `harness dump-comp`),
optimize with gristmill, export as a truthmill candidate Program JSON.

Usage: gristmill_baseline.py <comp.json> <out.json>
"""

import json
import sys

sys.path.insert(0, "/home/longli/code/gristmill-heuristics/tools")
import dspark_shim  # noqa: F401
from dummy_spark import SparkContext
from sympy import IndexedBase, Rational, Symbol
from drudge import PartHoleDrudge
from drudge.canonpy import Perm
from gristmill import optimize, get_flop_cost


def main():
    comp_path, out_path = sys.argv[1], sys.argv[2]
    comp = json.load(open(comp_path))

    dr = PartHoleDrudge(SparkContext())
    dr.full_simplify = False
    p = dr.names
    ranges = {0: p.O, 1: p.V}

    # Tensor bases + symmetries for inputs.
    bases = {}
    for tid_str, info in comp["tensors"].items():
        tid = int(tid_str)
        if tid >= 1000:
            continue  # outputs
        base = IndexedBase(f"T{tid}")
        bases[tid] = base
        gens = info.get("symmetry", [])
        if gens:
            perms = [
                Perm(g["perm"], 1 if g["action"] == "Negate" else 0)
                for g in gens
            ]
            dr.set_symm(base, perms, valence=info["rank"])

    def sym(i):
        return Symbol(f"x{i}")

    targets = []
    out_names = []
    for d in comp["definitions"]:
        base_id = d["base"]
        name = f"T{base_id}"
        out_names.append(name)
        terms = []
        for t in d["rhs"]:
            c = t["coeff"]
            if isinstance(c, str) and "/" in c:
                num, den = c.split("/")
                body = Rational(int(num), int(den))
            else:
                body = Rational(int(c)) if not isinstance(c, str) else Rational(int(c))
            for f in t["factors"]:
                body = body * bases[f["tensor"]][*[sym(i) for i in f["indices"]]]
            args = []
            for s in t["sums"]:
                args.append((sym(s["id"]), ranges[s["range"]]))
            args.append(body)
            terms.append(dr.sum(*args))
        from drudge import sum_ as dsum
        rhs = dsum(terms) if len(terms) > 1 else terms[0]
        base = IndexedBase(name)
        if d["exts"]:
            lhs_args = [base] + [(sym(e["id"]), ranges[e["range"]]) for e in d["exts"]]
        else:
            lhs_args = [base]
        targets.append(dr.define(*lhs_args, rhs))

    cost_in = get_flop_cost(targets)
    eval_seq = optimize(targets, substs={p.nv: p.no * 4}, interm_fmt="tau{}")
    cost_out = get_flop_cost(eval_seq)
    print("flops in:", cost_in)
    print("flops out:", cost_out)

    # ---- export as truthmill candidate ----
    values = []
    name_to_value = {}
    outputs = []

    def export(eval_seq):
        for td in eval_seq:
            lhs = td.lhs
            name = str(lhs.base) if hasattr(lhs, "base") else str(lhs)
            ext_list = list(td.exts)
            ext_dom = {idx: (0 if rng.label == "O" else 1) for idx, rng in ext_list}
            ext_order = [idx for idx, _ in ext_list]
            terms = td.rhs.terms.collect()
            term_value_ids = []
            for term in terms:
                dom_of = dict(ext_dom)
                for idx, rng in term.sums:
                    dom_of[idx] = 0 if rng.label == "O" else 1
                coeff, factors = term.amp.as_coeff_mul()
                coeff = Rational(coeff)
                label_of = {}
                domains = {}

                def lab(idx):
                    if idx not in label_of:
                        label_of[idx] = len(label_of)
                        domains[label_of[idx]] = dom_of[idx]
                    return label_of[idx]

                inputs = []
                args = []
                for f in factors:
                    if not hasattr(f, "indices"):
                        raise RuntimeError(f"unexpected factor {f}")
                    fname = str(f.base)
                    labels = [lab(ix) for ix in f.indices]
                    inputs.append(labels)
                    if fname.startswith("T"):
                        args.append({"Tensor": int(fname[1:])})
                    else:
                        args.append({"Value": name_to_value[fname]})
                out_labels = [lab(ix) for ix in ext_order]
                values.append({
                    "Einsum": {
                        "coeff": [coeff.p, coeff.q],
                        "code": {
                            "inputs": inputs,
                            "output": out_labels,
                            "domains": {str(k): v for k, v in domains.items()},
                        },
                        "args": args,
                    }
                })
                term_value_ids.append(len(values) - 1)
            if len(term_value_ids) == 1:
                final = term_value_ids[0]
            else:
                rank = len(ext_order)
                inputs = [list(range(rank)) for _ in term_value_ids]
                domains = {str(k): (0 if ext_list[k][1].label == "O" else 1) for k in range(rank)}
                values.append({
                    "LinComb": {
                        "coeff": [[1, 1] for _ in term_value_ids],
                        "code": {
                            "inputs": inputs,
                            "output": list(range(rank)),
                            "domains": domains,
                        },
                        "args": [{"Value": v} for v in term_value_ids],
                    }
                })
                final = len(values) - 1
            name_to_value[name] = final
            if name in out_names:
                outputs.append(final)

    export(eval_seq)
    prog = {"values": values, "outputs": outputs}
    with open(out_path, "w") as f:
        json.dump(prog, f)
    print("wrote", out_path, len(values), "values")


if __name__ == "__main__":
    main()
