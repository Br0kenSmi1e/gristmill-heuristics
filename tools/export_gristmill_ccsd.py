"""Derive CCD with drudge, optimize with gristmill, export the optimized
evaluation sequence as a truthmill candidate Program JSON."""

import json
import dspark_shim  # noqa: F401
from dummy_spark import SparkContext
from sympy import IndexedBase, Rational, factorial, Symbol, Indexed, Mul

from drudge import PartHoleDrudge, sum_, prod_
from gristmill import optimize, get_flop_cost

ctx = SparkContext()
dr = PartHoleDrudge(ctx)
dr.full_simplify = False
p = dr.names
c_, c_dag = p.c_, p.c_dag
v_dumms, o_dumms = p.V_dumms, p.O_dumms

t = IndexedBase("t")
dr.set_dbbar_base(t, 2)
corr = dr.einst(
    sum_(
        Rational(1, factorial(i) ** 2)
        * t[tuple(v_dumms[:i]) + tuple(o_dumms[:i])]
        * prod_(c_dag[j] for j in v_dumms[:i])
        * prod_(c_[j] for j in reversed(o_dumms[:i]))
        for i in [1, 2]
    )
)
curr = dr.ham
h_bar = dr.ham
for k in range(4):
    curr = (curr | corr).simplify() / (k + 1)
    h_bar += curr
h_bar = h_bar.simplify()
h_bar.repartition(cache=True)
en_eqn = h_bar.eval_fermi_vev().simplify()
dr.wick_parallel = 1
proj1 = c_dag[o_dumms[0]] * c_[v_dumms[0]]
t1_eqn = (proj1 * h_bar).eval_fermi_vev().simplify()
print("T1 done")
proj = prod_(c_dag[j] for j in o_dumms[:2]) * prod_(c_[j] for j in reversed(v_dumms[:2]))
t2_eqn = (proj * h_bar).eval_fermi_vev().simplify()
print("T2 done")

e = Symbol("e")
r = IndexedBase("r")
a, b = p.V_dumms[:2]
i, j = p.O_dumms[:2]
r1 = IndexedBase("r1")
targets = [
    dr.define(e, en_eqn),
    dr.define(r1, (a, p.V), (i, p.O), t1_eqn),
    dr.define(r, (a, p.V), (b, p.V), (i, p.O), (j, p.O), t2_eqn),
]
opt_seq = optimize(targets, substs={p.nv: p.no * 4}, interm_fmt="tau{}")
print("opt cost:", get_flop_cost(opt_seq))

# ---- translate to truthmill Program ----
# tensor name -> TensorId, matching truthmill ccd.json: 0=f, 1=g(u), 2=t2
TENSOR_IDS = {}  # resolved by (name, rank): ccsd.json: 0=f, 1=g(u), 2=t1, 3=t2
def tensor_id(base, rank):
    return {("f", 2): 0, ("u", 4): 1, ("t", 2): 2, ("t", 4): 3}[(base, rank)]
DOM = {"O": 0, "V": 1}

values = []          # list of ValueDef dicts
name_to_value = {}   # intermediate name -> ValueId
outputs = []         # (name, value id) for e and r


def index_range_map(term, exts):
    m = {}
    for idx, rng in term.sums:
        m[idx] = DOM[rng.label]
    for idx, rng in exts:
        m[idx] = DOM[rng.label]
    return m


for td in opt_seq:
    lhs = td.lhs
    name = str(lhs.base) if isinstance(lhs, Indexed) else str(lhs)
    ext_list = list(td.exts)  # list of (index, range) in lhs axis order
    ext_dom = {idx: DOM[rng.label] for idx, rng in ext_list}
    ext_order = [idx for idx, _ in ext_list]

    terms = td.rhs.terms.collect()
    term_value_ids = []
    for term in terms:
        dom_of = index_range_map(term, ext_list)
        coeff, factors = term.amp.as_coeff_mul()
        coeff = Rational(coeff)
        # local label ids
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
            if not isinstance(f, Indexed):
                raise RuntimeError(f"unexpected factor {f} in {term.amp}")
            base = str(f.base)
            labels = [lab(ix) for ix in f.indices]
            inputs.append(labels)
            if (base, len(f.indices)) in [("f", 2), ("u", 4), ("t", 2), ("t", 4)]:
                args.append({"Tensor": tensor_id(base, len(f.indices))})
            else:
                args.append({"Value": name_to_value[base]})
        out_labels = [lab(ix) for ix in ext_order]
        if not args:
            raise RuntimeError("constant term unsupported")
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
        # LinComb summing the term values (each already in lhs ext order)
        rank = len(ext_order)
        inputs = [list(range(rank)) for _ in term_value_ids]
        domains = {str(k): DOM[ext_list[k][1].label] for k in range(rank)}
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
    if name in ("e", "r1", "r"):
        outputs.append(final)

prog = {"values": values, "outputs": outputs}
with open("/home/longli/code/gristmill-heuristics/benches/candidates/ccsd_gristmill.json", "w") as f:
    json.dump(prog, f)
print("exported candidate:", len(values), "values, outputs", outputs)
