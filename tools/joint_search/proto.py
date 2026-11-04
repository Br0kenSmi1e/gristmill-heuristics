#!/usr/bin/env python3
"""Joint-search challenger prototype (stage 1).

ILP (HiGHS via scipy.optimize.milp) over sharing/factoring decisions on the
term-by-subterm structure + exact subset-DP parenthesizer. Pure producer of
truthmill Program (candidate) JSON; no harness changes.

Costs are exact FLOP counts: an einsum step costs 2 * prod(extents of all
labels it mentions) (one fused multiply-add per output element per summed
element). LinComb step costs len(args) * prod(output extents).

Subcommands:
  analyze  <problem.json> 0=16,1=64            term/collision statistics
  price    <candidate.json> 0=16,1=64          FLOP cost of a candidate
  solve    <problem.json> 0=16,1=64 <out.json> ILP+DP optimize each output,
                                               emit candidate JSON
"""

import json
import sys
from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
from itertools import combinations, permutations

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix


# ------------------------------------------------------------------ parsing

SYMS = {}


def parse_terms(problem, out_vid):
    """Fully normalized terms of the definition whose output is `out_vid`.

    The reference program may contain nested sums (a LinComb value used as a
    factor); these are distributed, expanding the term list exactly.
    Each term: coeff (Fraction), factors [(tensor_id, (labels..))], dom
    {label: domain_id}. Free labels are canonical output positions 0..r-1;
    summed labels are fresh ids >= r (term-local)."""
    values = problem["reference"]["values"]
    top = values[out_vid]
    code = top["LinComb"]["code"] if "LinComb" in top else top["Einsum"]["code"]
    r = len(code["output"])

    nxt = [r]          # fresh dummy label counter
    dom = {}           # canonical label -> domain id (per definition)

    def expand_value(vid, pos2canon):
        """-> list of (coeff, [(tensor_id, canonical labels)]) terms."""
        v = values[vid]
        if "LinComb" in v:
            lc = v["LinComb"]
            out_q = {lab: q for q, lab in enumerate(lc["code"]["output"])}
            res = []
            for cf, inp, arg in zip(lc["coeff"], lc["code"]["inputs"],
                                    lc["args"]):
                c = Fraction(*cf)
                # child output axis p sits at canonical position
                # pos2canon[out_q[inp[p]]]
                child_map = {p: pos2canon[out_q[inp[p]]]
                             for p in range(len(inp))}
                if "Tensor" in arg:
                    for p in range(len(inp)):
                        dom.setdefault(child_map[p],
                                       lc["code"]["domains"][str(inp[p])])
                    res.append((c, [(arg["Tensor"], tuple(
                        child_map[p] for p in range(len(inp))))]))
                else:
                    res.extend((c * sc, fs)
                               for sc, fs in
                               expand_value(arg["Value"], child_map))
            return res
        e = v["Einsum"]
        ec = e["code"]
        out_pos = {idx: pos2canon[p] for p, idx in enumerate(ec["output"])}
        local = {}

        def lab(i):
            if i in out_pos:
                l = out_pos[i]
            else:
                if i not in local:
                    local[i] = nxt[0]
                    nxt[0] += 1
                l = local[i]
            dom[l] = ec["domains"][str(i)]
            return l

        res = [(Fraction(*e["coeff"]), [])]
        for arg, inp in zip(e["args"], ec["inputs"]):
            if "Tensor" in arg:
                res = [(c, fs + [(arg["Tensor"], tuple(lab(i) for i in inp))])
                       for c, fs in res]
            else:
                sub = expand_value(arg["Value"],
                                   {p: lab(i) for p, i in enumerate(inp)})
                res = [(c * sc, fs + sfs)
                       for c, fs in res for sc, sfs in sub]
        return res

    terms = []
    for c, fs in expand_value(out_vid, {q: q for q in range(r)}):
        used = {l for _, ls in fs for l in ls} | set(range(r))
        terms.append({"coeff": c, "factors": fs,
                      "dom": {l: dom[l] for l in used}, "rank": r})
    return terms, r


# ------------------------------------------------------------- FLOP pricing

def step_cost(in_label_sets, out_label_set, dom, ext):
    """truthmill FLOP convention: (n_args - 1 + has_contraction) times the
    product of extents over all labels the step mentions."""
    labs = set(out_label_set)
    for s in in_label_sets:
        labs |= set(s)
    n_args = len(in_label_sets)
    has_contr = len(out_label_set) < len(labs)
    p = 1
    for l in labs:
        p *= ext[dom[l]]
    return (n_args - 1 + (1 if has_contr else 0)) * p


def einsum_cost(label_domains, ext):
    p = 1
    for d in label_domains:
        p *= ext[d]
    return 2 * p


def dp_tree(factors, out_labels, dom, ext):
    """Optimal binary parenthesization of an einsum over `factors`
    (list of label tuples), keeping `out_labels`. Returns (cost, tree) with
    tree = int (leaf index) | (ltree, rtree); and result labels per node are
    recomputed by `node_labels`."""
    n = len(factors)
    lab_of = [frozenset(f) for f in factors]
    out = frozenset(out_labels)

    def result_labels(mask):
        inside = frozenset().union(*(lab_of[i] for i in range(n)
                                     if (mask >> i) & 1))
        outside = frozenset().union(*(lab_of[i] for i in range(n)
                                      if not (mask >> i) & 1)) \
            if mask != (1 << n) - 1 else frozenset()
        return inside & (out | outside)

    @lru_cache(maxsize=None)
    def best(mask):
        if mask & (mask - 1) == 0:
            return 0, None
        bc, bsplit = None, None
        sub = (mask - 1) & mask
        while sub:
            other = mask ^ sub
            if other and sub < other:
                c1, _ = best(sub)
                c2, _ = best(other)
                la = result_labels(sub)
                lb = result_labels(other)
                step = step_cost([la, lb], result_labels(mask), dom, ext)
                tot = c1 + c2 + step
                if bc is None or tot < bc:
                    bc, bsplit = tot, (sub, other)
            sub = (sub - 1) & mask
        return bc, bsplit

    full = (1 << n) - 1

    def build(mask):
        if mask & (mask - 1) == 0:
            return mask.bit_length() - 1
        _, (a, b) = best(full if False else mask)
        return (build(a), build(b))

    cost, _ = best(full)
    return cost, build(full), result_labels


# ------------------------------------------------- canonical subterm forms

def sym_orbit_min(tid, labels, syms):
    """Reduce a factor's label tuple modulo the tensor's declared signed
    permutation symmetries: return (min label tuple over the orbit, sign of
    the chosen representative relative to the input)."""
    gens = syms.get(str(tid), [])
    if not gens:
        return tuple(labels), 1
    import itertools
    best, bestsign = None, 1
    # orbit = group closure of generators (small; brute force)
    orbit = {(tuple(labels), 1)}
    frontier = [(tuple(labels), 1)]
    while frontier:
        cur, sgn = frontier.pop()
        for g in gens:
            perm, s = g["permutation"], g["sign"]
            nxt = tuple(cur[perm[i]] for i in range(len(cur)))
            ns = sgn * s
            if (nxt, ns) not in orbit:
                orbit.add((nxt, ns))
                frontier.append((nxt, ns))
    for tup, sgn in orbit:
        if best is None or tup < best:
            best, bestsign = tup, sgn
    return best, bestsign


def canon_sig(factors, dom, free, want_map=False, syms=None):
    """Canonical isomorphism signature of a subterm: invariant under factor
    permutation and label bijection, with (domain, free/summed) attributes
    attached per label. If SYMS (tensor_id -> generator list) is set,
    factors are first reduced modulo their declared symmetries (sign ignored
    in the signature; emission must track it separately)."""
    if SYMS:
        factors = [(tid, sym_orbit_min(tid, ls, SYMS)[0])
                   for tid, ls in factors]
    labs = sorted({l for _, ls in factors for l in ls})
    by_attr = defaultdict(list)
    for l in labs:
        by_attr[(dom[l], l in free)].append(l)
    groups = sorted(by_attr.items())          # [((dom,free), [labels])]
    per_group = [list(permutations(ls)) for _, ls in groups]
    attrs = tuple(a for a, ls in groups for _ in ls)

    best = None
    bestren = None
    seen = set()
    for fp in permutations(range(len(factors))):
        tids = tuple(factors[i][0] for i in fp)
        if tids in seen:
            continue
        seen.add(tids)
        pf = [factors[i] for i in fp]
        for combo in _product(per_group):
            ren = {}
            i = 0
            for labels_in_group in combo:
                for l in labels_in_group:
                    ren[l] = i
                    i += 1
            body = tuple(sorted(
                (tid, tuple(ren[l] for l in ls)) for tid, ls in pf))
            sig = (attrs, body)
            if best is None or sig < best:
                best, bestren = sig, ren
    if want_map:
        return best, bestren
    return best


def _product(lists):
    if not lists:
        yield []
        return
    for x in lists[0]:
        for xs in _product(lists[1:]):
            yield [x] + xs


def subterm_pools(terms):
    """All factor subsets of size>=2 per term, with canonical signatures.
    Returns pools[t] = [(frozenset(factor idxs), sig, free_labels)]."""
    pools = []
    for t, term in enumerate(terms):
        k = len(term["factors"])
        pool = []
        for size in range(2, k + 1):
            for idxs in combinations(range(k), size):
                idxs = frozenset(idxs)
                facs = [term["factors"][i] for i in idxs]
                labs = {l for _, ls in facs for l in ls}
                # free = appears outside the subset or in the term output
                outside = set()
                for j, (_, ls) in enumerate(term["factors"]):
                    if j not in idxs:
                        outside |= set(ls)
                outside |= set(range(term["rank"]))
                free = labs & outside
                sig = canon_sig(facs, term["dom"], free)
                pool.append((idxs, sig, frozenset(free)))
        pools.append(pool)
    return pools


# ------------------------------------------------- shared subterm analysis

def term_groups(terms):
    """Group terms equal up to an output-axis permutation. Returns
    (groups, perms): groups = [(tsig, [member term indices])], first member
    is the representative; perms[t][p] = representative output position
    corresponding to member t's output position p."""
    groups = {}
    rens = {}
    for t, term in enumerate(terms):
        free = set(range(term["rank"]))
        sig, ren = canon_sig(term["factors"], term["dom"], free,
                             want_map=True)
        rens[t] = ren
        groups.setdefault(sig, []).append(t)
    perms = {}
    for sig, members in groups.items():
        rep = members[0]
        ren_rep_inv = {v: k for k, v in rens[rep].items()}
        for t in members:
            r = terms[t]["rank"]
            perms[t] = [ren_rep_inv[rens[t][p]] for p in range(r)]
    return [(sig, members) for sig, members in groups.items()], perms


def shared_candidates(terms):
    """Find shareable subterms (canonical sigs occurring in >=2 terms).
    Returns (entries, siginfo):
      entries[t] = [(factor_idx frozenset, sig, free frozenset, ren dict)]
      siginfo[sig] = {"cfactors": canonical factors, "cfree": canonical free
                      ids, "cdom": canonical label -> domain}"""
    pools = []
    sig_terms = defaultdict(set)
    for t, term in enumerate(terms):
        k = len(term["factors"])
        pool = []
        for size in range(2, k + 1):
            for idxs in combinations(range(k), size):
                idxs = frozenset(idxs)
                facs = [term["factors"][i] for i in idxs]
                labs = {l for _, ls in facs for l in ls}
                outside = set(range(term["rank"]))
                for j, (_, ls) in enumerate(term["factors"]):
                    if j not in idxs:
                        outside |= set(ls)
                free = labs & outside
                sig, ren = canon_sig(facs, term["dom"], free, want_map=True)
                pool.append((idxs, sig, free, ren))
                sig_terms[sig].add(t)
        pools.append(pool)
    shared = {s for s, ts in sig_terms.items() if len(ts) >= 2}
    entries = []
    siginfo = {}
    for t, (term, pool) in enumerate(zip(terms, pools)):
        ent = []
        for idxs, sig, free, ren in pool:
            if sig in shared:
                ent.append((idxs, sig, free, ren))
                if sig not in siginfo:
                    facs = [term["factors"][i] for i in idxs]
                    cfactors = tuple(sorted(
                        (tid, tuple(ren[l] for l in ls)) for tid, ls in facs))
                    siginfo[sig] = {
                        "cfactors": cfactors,
                        "cfree": tuple(sorted(ren[l] for l in free)),
                        "cdom": {i: a[0] for i, a in enumerate(sig[0])},
                    }
        entries.append(ent)
    return entries, siginfo


def disjoint_subsets(ent):
    """All pairwise-factor-disjoint subsets of a term's shareable entries."""
    out = [()]
    for e in ent:
        for s in list(out):
            if all(not (e[0] & x[0]) for x in s):
                out.append(s + (e,))
    return out


# ------------------------------------------------------------------ the ILP

def solve_definition(terms, ext, cap=None):
    """One definition (one output). Returns None or
    (obj, chosen, siginfo, groups, perms); chosen[rep_t] =
    (subset, grouped_sig_or_None), keyed by representative term index."""
    groups, perms = term_groups(terms)
    rep_ids = [m[0] for _, m in groups]
    rep_terms = [terms[t] for t in rep_ids]
    entries, siginfo = shared_candidates(rep_terms)
    sigs = sorted(siginfo)
    x_idx = {s: i for i, s in enumerate(sigs)}

    icost = {}
    for s in sigs:
        info = siginfo[s]
        cfacs = [list(ls) for _, ls in info["cfactors"]]
        cst, _, _ = dp_tree(cfacs, info["cfree"], info["cdom"], ext)
        icost[s] = cst

    def out_size(term):
        p = 1
        for l in range(term["rank"]):
            p *= ext[term["dom"][l]]
        return p

    # per-term choices: (t, subset, grouped_sig_or_None, cost)
    zvars = []
    for t, term in enumerate(rep_terms):
        osize = out_size(term)
        for subset in disjoint_subsets(entries[t]):
            used = frozenset().union(*(e[0] for e in subset)) if subset \
                else frozenset()
            singles = [list(ls) for i, (_, ls) in enumerate(term["factors"])
                       if i not in used]
            # ungrouped: all extracted sigs are atoms
            atoms = [sorted(e[2]) for e in subset] + singles
            cst, _, _ = dp_tree(atoms, list(range(term["rank"])),
                                term["dom"], ext)
            zvars.append((t, subset, None, cst))
            # grouped variants: one sig factors out; the rest sums first
            for e in subset:
                g = e[1]
                rest = [sorted(x[2]) for x in subset if x[1] != g] + singles
                rest_out = sorted(e[2]) + list(range(term["rank"]))
                cst, _, _ = dp_tree(rest, rest_out, term["dom"], ext)
                add = 1
                for l in rest_out:
                    add *= ext[term["dom"][l]]
                # -osize: this term no longer occupies a final-lincomb arg
                zvars.append((t, subset, g, cst + add - osize))

    nz = len(zvars)
    xsigs = sigs
    nx = len(xsigs)
    wbase = nz + nx
    nw = len(sigs)
    n = nz + nx + nw

    c = np.zeros(n)
    for j, (_, _, _, cst) in enumerate(zvars):
        c[j] = zvars[j][3]
    for i, s in enumerate(xsigs):
        c[nz + i] = icost[s]
    for i, s in enumerate(sigs):
        info = siginfo[s]
        r0 = terms[0]["rank"]
        p = 1
        for l in info["cfree"]:
            p *= ext[info["cdom"][l]]
        for l in range(r0):
            p *= ext[terms[0]["dom"][l]]
        osize = 1
        for l in range(r0):
            osize *= ext[terms[0]["dom"][l]]
        # group cost: materialize g once + one combine g x S + one lincomb arg
        c[wbase + i] = icost[s] + 2 * p + osize
    integrality = np.ones(n)
    bounds = Bounds(np.zeros(n), np.ones(n))

    rows, lb, ub = [], [], []
    for t in range(len(rep_terms)):
        row = lil_matrix((1, n))
        for j, (zt, _, _, _) in enumerate(zvars):
            if zt == t:
                row[0, j] = 1.0
        rows.append(row)
        lb.append(1.0)
        ub.append(1.0)
    for j, (t, subset, grouped, _) in enumerate(zvars):
        for e in subset:
            row = lil_matrix((1, n))
            row[0, j] = 1.0
            if e[1] == grouped:
                row[0, wbase + x_idx[e[1]]] = -1.0  # needs w_g
            else:
                row[0, nz + x_idx[e[1]]] = -1.0     # needs x_g
            rows.append(row)
            lb.append(-np.inf)
            ub.append(0.0)
    if cap is not None:
        row = lil_matrix((1, n))
        for j in range(n):
            row[0, j] = c[j]
        rows.append(row)
        lb.append(0.0)
        ub.append(cap)

    from scipy.sparse import vstack
    A = vstack(rows).tocsr()
    res = milp(c=c, constraints=LinearConstraint(A, lb, ub),
               integrality=integrality, bounds=bounds,
               options={"time_limit": 300})
    if res.x is None:
        return None
    xs = res.x
    chosen = {}
    for j, (t, subset, grouped, _) in enumerate(zvars):
        if xs[j] > 0.5:
            chosen[rep_ids[t]] = (subset, grouped)
    return res.fun, chosen, siginfo, groups, perms


# ---------------------------------------------------------------- emission

def emit_binary_tree(values, leaf_info, out_order, dom, ext):
    """Emit a binary contraction tree over `leaf_info` = [(argref_dict,
    labels tuple in that arg's own axis order)]. The root's declared output
    order is `out_order`; internal nodes keep labels needed later, sorted.
    Returns (argref of result, output labels tuple)."""
    if len(leaf_info) == 1:
        ref, ls = leaf_info[0]
        if tuple(ls) == tuple(out_order):
            return ref, tuple(out_order)
        vid = len(values)
        labs = sorted(set(ls) | set(out_order))
        values.append({"Einsum": {
            "coeff": [1, 1],
            "code": {"inputs": [list(ls)], "output": list(out_order),
                     "domains": {str(l): dom[l] for l in labs}},
            "args": [ref]}})
        return {"Value": vid}, tuple(out_order)

    _, tree, _ = dp_tree([list(ls) for _, ls in leaf_info],
                         list(out_order), dom, ext)

    def emit(node, is_root):
        if isinstance(node, int):
            ref, ls = leaf_info[node]
            return ref, tuple(ls), {node}
        refa, la, seta = emit(node[0], False)
        refb, lb, setb = emit(node[1], False)
        inside = set(la) | set(lb)
        outside = set(out_order)
        for i, (_, ls) in enumerate(leaf_info):
            if i not in (seta | setb):
                outside |= set(ls)
        keep = sorted(inside & outside)
        out = list(out_order) if is_root else keep
        vid = len(values)
        labs = sorted(set(la) | set(lb) | set(out))
        values.append({"Einsum": {
            "coeff": [1, 1],
            "code": {"inputs": [list(la), list(lb)], "output": out,
                     "domains": {str(l): dom[l] for l in labs}},
            "args": [refa, refb]}})
        return {"Value": vid}, tuple(out), seta | setb

    ref, out, _ = emit(tree, True)
    return ref, out


S_BASE = 100000  # namespace for group-sum intermediate labels


def emit_program(problem, ext):
    """Full pipeline: per output definition, ILP+DP, emit one Program."""
    values = []
    outputs = []
    report = []
    for ov in problem["reference"]["outputs"]:
        terms, r = parse_terms(problem, ov)
        res = solve_definition(terms, ext)
        if res is None:
            raise RuntimeError(f"ILP failed on output {ov}")
        obj, chosen, siginfo, groups, perms = res
        members_of = {}   # rep term index -> [member term indices]
        for sig, members in groups:
            members_of[members[0]] = members

        indiv, grouped = set(), defaultdict(list)
        for rep_t, (subset, g) in chosen.items():
            for e in subset:
                if e[1] == g:
                    grouped[g].append(rep_t)
                else:
                    indiv.add(e[1])

        # materialize shared intermediates (individual + group factors)
        sig_value = {}
        for s in sorted(indiv | set(grouped)):
            info = siginfo[s]
            leaf_info = [({"Tensor": tid}, ls) for tid, ls in
                         info["cfactors"]]
            ref, out = emit_binary_tree(values, leaf_info,
                                        list(info["cfree"]), info["cdom"],
                                        ext)
            sig_value[s] = (ref, out)

        def emit_rest(rep_t, out_order):
            term = terms[rep_t]
            subset, g = chosen[rep_t]
            usedf = frozenset().union(*(e[0] for e in subset)) if subset \
                else frozenset()
            leaf_info = []
            for e in subset:
                if e[1] == g:
                    continue
                _, sig, free, ren = e
                ref, order = sig_value[sig]
                ren_inv = {v: k for k, v in ren.items()}
                leaf_info.append((ref, tuple(ren_inv[c] for c in order)))
            for i, (tid, ls) in enumerate(term["factors"]):
                if i not in usedf:
                    leaf_info.append(({"Tensor": tid}, tuple(ls)))
            return emit_binary_tree(values, leaf_info, out_order,
                                    term["dom"], ext)

        contribs = []   # (argref, coeff) for the final lincomb

        # factor-out groups: S = sum of (permuted) rests, then one combine
        for g, reps in grouped.items():
            info = siginfo[g]
            cfree = list(info["cfree"])
            s_order = [S_BASE + c for c in cfree] + list(range(r))
            s_dom = {str(S_BASE + c): info["cdom"][c] for c in cfree}
            for l in range(r):
                s_dom[str(l)] = terms[0]["dom"][l]
            sum_args, sum_coeffs = [], []
            for rep_t in reps:
                ge = [e for e in chosen[rep_t][0] if e[1] == g][0]
                ren_inv = {v: k for k, v in ge[3].items()}
                rest_out = [ren_inv[c] for c in cfree] + list(range(r))
                ref, _ = emit_rest(rep_t, rest_out)
                # members: permute rest over output positions
                for t in members_of[rep_t]:
                    pi = perms[t]
                    if all(p == q for p, q in enumerate(pi)):
                        sum_args.append(ref)
                    else:
                        vid = len(values)
                        values.append({"Einsum": {
                            "coeff": [1, 1],
                            "code": {
                                "inputs": [s_order],
                                "output": [S_BASE + c for c in cfree]
                                          + [0] * 0 + [pi[p] for p in range(r)],
                                "domains": dict(s_dom)},
                            "args": [ref]}})
                        sum_args.append({"Value": vid})
                    sum_coeffs.append(terms[t]["coeff"])
            vid = len(values)
            values.append({"LinComb": {
                "coeff": [[c.numerator, c.denominator] for c in sum_coeffs],
                "code": {"inputs": [s_order] * len(sum_args),
                         "output": s_order, "domains": s_dom},
                "args": sum_args}})
            gref, gorder = sig_value[g]
            vid2 = len(values)
            values.append({"Einsum": {
                "coeff": [1, 1],
                "code": {"inputs": [[S_BASE + c for c in gorder], s_order],
                         "output": list(range(r)),
                         "domains": dict(s_dom)},
                "args": [gref, {"Value": vid}]}})
            contribs.append(({"Value": vid2}, Fraction(1)))

        # ungrouped reps: evaluate once, permute per member
        for rep_t, (subset, g) in chosen.items():
            if g is not None:
                continue
            ref, _ = emit_rest(rep_t, list(range(r)))
            for t in members_of[rep_t]:
                pi = perms[t]
                if all(p == q for p, q in enumerate(pi)):
                    contribs.append((ref, terms[t]["coeff"]))
                else:
                    vid = len(values)
                    outdom = {str(l): terms[t]["dom"][l] for l in range(r)}
                    values.append({"Einsum": {
                        "coeff": [1, 1],
                        "code": {"inputs": [list(range(r))],
                                 "output": [pi[p] for p in range(r)],
                                 "domains": outdom},
                        "args": [ref]}})
                    contribs.append(({"Value": vid}, terms[t]["coeff"]))

        outdom = {str(l): terms[0]["dom"][l] for l in range(r)}
        vid = len(values)
        values.append({"LinComb": {
            "coeff": [[cf.numerator, cf.denominator] for _, cf in contribs],
            "code": {"inputs": [list(range(r))] * len(contribs),
                     "output": list(range(r)),
                     "domains": outdom},
            "args": [ref for ref, _ in contribs]}})
        outputs.append(vid)
        report.append((ov, len(terms), len(groups), int(obj)))
    return {"values": values, "outputs": outputs}, report


# --------------------------------------------------------------- candidate

def price_candidate(cand, ext):
    """truthmill log_flops convention, exact integer total."""
    tot = 0
    for v in cand["values"]:
        if "Einsum" in v:
            op = v["Einsum"]
            code = op["code"]
            has_contr = len(code["output"]) < len(code["domains"])
            ops = len(op["args"]) - 1 + (1 if has_contr else 0)
            if ops:
                p = 1
                for d in code["domains"].values():
                    p *= ext[d]
                tot += ops * p
        else:
            op = v["LinComb"]
            ops = len(op["args"]) - 1
            if ops:
                p = 1
                for i in op["code"]["output"]:
                    p *= ext[op["code"]["domains"][str(i)]]
                tot += ops * p
    return tot


def naive_cost(terms, ext, out_size=0):
    """Reference-style cost: each term as one n-ary einsum step under the
    truthmill convention: (k - 1 + has_contraction) * prod(all extents)."""
    tot = 0
    for t in terms:
        labs = {l for _, ls in t["factors"] for l in ls}
        p = 1
        for l in labs:
            p *= ext[t["dom"][l]]
        has_contr = len(labs - set(range(t["rank"]))) > 0
        tot += (len(t["factors"]) - 1 + (1 if has_contr else 0)) * p
    return tot + max(0, len(terms) - 1) * out_size


# ------------------------------------------------------------------- driver

def cmd_analyze(prob_path, sizes):
    problem = json.load(open(prob_path))
    ref = problem["reference"]
    for ov in ref["outputs"]:
        terms, r = parse_terms(problem, ov)
        print(f"output {ov}: {len(terms)} terms, rank {r}")
        kc = defaultdict(int)
        for t in terms:
            kc[len(t["factors"])] += 1
        print("  factors/term:", dict(sorted(kc.items())))
        pools = subterm_pools(terms)
        sig_count = defaultdict(set)
        for t, pool in enumerate(pools):
            for _, sig, _ in pool:
                sig_count[sig].add(t)
        shared = {s: ts for s, ts in sig_count.items() if len(ts) >= 2}
        print(f"  subterms: {sum(len(p) for p in pools)}, "
              f"distinct sigs: {len(sig_count)}, shared across terms: "
              f"{len(shared)}")
        if shared:
            use = sorted((len(ts) for ts in shared.values()), reverse=True)
            print(f"  sharing fan-out (top): {use[:8]}")


def main():
    cmd = sys.argv[1]
    if cmd == "analyze":
        cmd_analyze(sys.argv[2], parse_sizes(sys.argv[3]))
    elif cmd == "price":
        cand = json.load(open(sys.argv[2]))
        ext = parse_sizes(sys.argv[3])
        print(f"{sys.argv[2]}: {price_candidate(cand, ext)} FLOPs")
    elif cmd == "solve":
        prob_path, sizes, out_path = sys.argv[2], parse_sizes(sys.argv[3]), \
            sys.argv[4]
        problem = json.load(open(prob_path))
        prog, report = emit_program(problem, sizes)
        priced = price_candidate(prog, sizes)
        for ov, n, ng, obj in report:
            print(f"output {ov}: {n} terms ({ng} permutation groups) | "
                  f"ilp objective {obj}")
        json.dump(prog, open(out_path, "w"))
        print(f"emitted {len(prog['values'])} steps -> {out_path}")
        print(f"emitted program priced: {priced} FLOPs "
              f"(log10 = {np.log10(priced):.6f})")
    else:
        print(__doc__)
        sys.exit(1)


def sigs_used(subsets):
    return {e[1] for s in subsets for e in s}


def parse_sizes(s):
    return {int(k): int(v) for k, v in
            (kv.split("=") for kv in s.split(","))}


if __name__ == "__main__":
    main()
