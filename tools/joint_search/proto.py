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
import time as time_mod
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

def sym_orbit(tid, labels, syms):
    """The full signed orbit of a factor's label tuple under the tensor's
    declared signed permutation symmetries: list of (tuple, sign)."""
    gens = syms.get(str(tid), [])
    if not gens:
        return [(tuple(labels), 1)]
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
    return sorted(orbit)


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


def canon_full(factors, dom, free, syms=None):
    """Returns (sig, ren, sign): canonical isomorphism signature, label
    renaming (original -> canonical id), and the product of per-factor
    orbit signs (instance = sign x canonical, as functions).

    The min is over label bijections x per-factor signed symmetry orbits
    (C-015: mining in the original label space first is NOT canonical).
    Exact and fast: the body is a SORTED multiset, and order statistics
    are monotone per element, so for a fixed renaming the per-factor
    orbit-min in canonical space is the exact joint orbit-min. (Factor
    order is irrelevant to the sorted body; the old factor-permutation
    loop was dead work.) Ties with opposite signs imply a signed
    automorphism, i.e. a zero network, so either choice is sound."""
    if SYMS:
        orbits = [sym_orbit(tid, ls, SYMS) for tid, ls in factors]
    else:
        orbits = [[(tuple(ls), 1)] for _, ls in factors]
    labs = sorted({l for _, ls in factors for l in ls})
    by_attr = defaultdict(list)
    for l in labs:
        by_attr[(dom[l], l in free)].append(l)
    groups = sorted(by_attr.items())          # [((dom,free), [labels])]
    per_group = [list(permutations(ls)) for _, ls in groups]
    attrs = tuple(a for a, ls in groups for _ in ls)

    best = None
    bestren = None
    bestsign = 1
    for combo in _product(per_group):
        ren = {}
        i = 0
        for labels_in_group in combo:
            for l in labels_in_group:
                ren[l] = i
                i += 1
        body_items = []
        sign = 1
        for (tid, _), orb in zip(factors, orbits):
            bt = None
            bs = 1
            for tup, s in orb:
                rt = tuple(ren[l] for l in tup)
                if bt is None or rt < bt:
                    bt, bs = rt, s
            body_items.append((tid, bt))
            sign *= bs
        sig = (attrs, tuple(sorted(body_items)))
        if best is None or sig < best:
            best, bestren, bestsign = sig, ren, sign
    return best, bestren, bestsign


def canon_sig(factors, dom, free, want_map=False, syms=None):
    out = canon_full(factors, dom, free)
    if want_map:
        return out[0], out[1]
    return out[0]


def _canon_sig_core(factors, dom, free):
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
    return best, bestren


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
    signs = {}
    for t, term in enumerate(terms):
        free = set(range(term["rank"]))
        sig, ren, sgn = canon_full(term["factors"], term["dom"], free)
        rens[t] = ren
        signs[t] = sgn
        groups.setdefault(sig, []).append(t)
    perms = {}
    for sig, members in groups.items():
        rep = members[0]
        ren_rep_inv = {v: k for k, v in rens[rep].items()}
        for t in members:
            r = terms[t]["rank"]
            perms[t] = [ren_rep_inv[rens[t][p]] for p in range(r)]
    return [(sig, members) for sig, members in groups.items()], perms, signs


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


# ------------------------------------------------- global multi-level ILP
#
# Universe lemma: every intermediate in any valid schedule is a factor subset
# of some term (up to wiring isomorphism), with a free-label set determined by
# a use context (a term, or another intermediate's formation). The universe U
# is the closure of the term-level subsets under that context rule. The ILP
# below is exact over the full product-sharing DAG space: x_s materializes a
# sig, y_{M,S} chooses its formation (disjoint cover by materialized blocks +
# singleton leftovers, DP-priced), z_{t,S} likewise for terms.

def canon_record(sig):
    """Unpack a canon_sig into (cfactors, cfree, cdom)."""
    attrs, body = sig
    cfree = tuple(i for i, a in enumerate(attrs) if a[1])
    cdom = {i: a[0] for i, a in enumerate(attrs)}
    return list(body), cfree, cdom


def subset_sigs(cfactors, cdom, cfree):
    """All proper factor subsets (size>=2) of a canonical record, each with
    its context free set, canonicalized. Yields (frozenset idxs, sig,
    free_in_context, ren, sgn) — free/ren in the record's own label
    namespace; sgn = orbit sign of the block instance vs its canonical
    form (instance = sgn x canonical). NOTE: mining does not commute with
    canonical relabeling, so sgn can be -1 even though the parent's
    factors were orbit-mined in their original term namespace (B-006)."""
    n = len(cfactors)
    for size in range(2, n):
        for idxs in combinations(range(n), size):
            idxs = frozenset(idxs)
            facs = [cfactors[i] for i in idxs]
            labs = {l for _, ls in facs for l in ls}
            outside = set(cfree)
            for j, (_, ls) in enumerate(cfactors):
                if j not in idxs:
                    outside |= set(ls)
            free = labs & outside
            sig, ren, sgn = canon_full(facs, cdom, free)
            yield idxs, sig, free, ren, sgn


def build_universe(terms):
    """Closure of term-level factor subsets under context freeness.
    Returns (U, term_blocks, term_entries) with U: sig -> record;
    term_blocks[t] = [(frozenset idxs, sig)] all subset sigs of term t;
    term_entries[t] = [(frozenset idxs, sig, free, ren)] with maps."""
    U = {}
    work = []
    term_blocks = []
    term_entries = []
    for term in terms:
        k = len(term["factors"])
        blocks = []
        entries = []
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
                sig, ren, sgn = canon_full(facs, term["dom"], free)
                blocks.append((idxs, sig))
                entries.append((idxs, sig, free, ren, sgn))
                if sig not in U:
                    work.append(sig)
        term_blocks.append(blocks)
        term_entries.append(entries)
    while work:
        sig = work.pop()
        if sig in U:
            continue
        cfactors, cfree, cdom = canon_record(sig)
        U[sig] = {"cfactors": cfactors, "cfree": cfree, "cdom": cdom}
        for _, subsig, _, _, _ in subset_sigs(cfactors, cdom, cfree):
            if subsig not in U:
                work.append(subsig)
    return U, term_blocks, term_entries


def cover_choices(atom_sets, block_list, out_labels, dom, ext):
    """All ways to cover the term/formation by disjoint candidate blocks;
    returns [(frozenset-of-block-positions, dp_cost)]."""
    choices = []
    n = len(atom_sets)
    for cover in disjoint_subsets(block_list):
        used = frozenset().union(*(b[0] for b in cover)) if cover \
            else frozenset()
        atoms = [sorted(b[2]) for b in cover]
        for i in range(n):
            if i not in used:
                atoms.append(sorted(atom_sets[i]))
        cst, _, _ = dp_tree(atoms, list(out_labels), dom, ext)
        choices.append((cover, cst))
    return choices


def sum_group_candidates(terms, term_entries, partial_keep=True,
                         locked=None):
    """Group candidates = cofactor families: (canonical cofactor sig,
    canonical rest-carry split). The cofactor may carry some output labels
    (partial keep); the S space is (canonical g-free labels present in
    rests) + (out positions present in rests), consistent within a key."""
    groups = defaultdict(list)   # key -> [(t, idxs)]
    for t, term in enumerate(terms):
        k = len(term["factors"])
        r = term["rank"]
        outpos = set(range(r))
        for size in range(1, k):
            for idxs in combinations(range(k), size):
                idxs = frozenset(idxs)
                facs = [term["factors"][i] for i in idxs]
                rest = [term["factors"][i] for i in range(k)
                        if i not in idxs]
                labs_g = {l for _, ls in facs for l in ls}
                rest_labels = {l for _, ls in rest for l in ls}
                full = ((labs_g & (rest_labels | outpos)) <= rest_labels
                        and outpos <= rest_labels)
                if not partial_keep and not full:
                    continue
                # partial-keep families are sound only for identity-perm
                # members; locked reps (those with permuted siblings) are
                # restricted to full-keep families
                if locked and t in locked and not full:
                    continue
                free_g = labs_g & (rest_labels | outpos)
                sig, ren, sgn = canon_full(facs, term["dom"], free_g)
                ren_inv = {v: kk for kk, v in ren.items()}
                cf_rest = tuple(sorted(c for c in ren_inv
                                       if ren_inv[c] in rest_labels))
                out_g = tuple(sorted(p for p in range(r)
                                     if p in labs_g
                                     and p not in rest_labels))
                key = (sig, cf_rest, out_g)
                groups[key].append((t, idxs, ren, sgn))
    return {key: ctx for key, ctx in groups.items() if len(ctx) >= 2}


def orbit_signature(terms, members, perms, tsigns):
    """Orbit signature of one term group: (c_eff, sig) where c_eff is the
    rep coeff with identity-perm members folded in, and sig = sorted tuple
    of (perm tuple, ratio) for non-identity members, ratio = member's
    needed coeff / c_eff. Two groups with equal sig can share one pre-sum
    whose orbit rides it (C-012); a family whose reps all share one sig
    can be ridden as a whole (C-016)."""
    rep = members[0]
    c_eff = terms[rep]["coeff"]
    nonid = []
    for t in members[1:]:
        pi = perms[t]
        if all(p == q for p, q in enumerate(pi)):
            c_eff += terms[t]["coeff"] * tsigns[t] * tsigns[rep]
        else:
            nonid.append((tuple(pi),
                          terms[t]["coeff"] * tsigns[t] * tsigns[rep]))
    if c_eff == 0:
        return c_eff, None   # degenerate (rep cancels); never cluster
    sig = tuple(sorted((pi, rho / c_eff) for pi, rho in nonid))
    return c_eff, sig


def sum_orbit_clusters(terms, groups, perms, tsigns):
    """C-012: cluster reps whose non-identity member (perm, coeff-ratio)
    signatures are identical, so one pre-sum serves the whole orbit:
    (A + B) + pi(A + B) instead of A + pi(A) + B + pi(B).
    Only groups of the form {rep + non-identity members} are eligible
    (identity members fold into the rep coeff instead). Returns a list of
    clusters; each cluster is ([rep local ids], [perms], [ratios]).
    Disjoint: a rep joins at most one cluster (first signature wins)."""
    sig_map = {}
    for _, members in groups:
        rep = members[0]
        _c_eff, sig = orbit_signature(terms, members, perms, tsigns)
        if not sig:
            continue
        sig_map.setdefault(sig, []).append(rep)
    clusters = []
    used = set()
    for sig, cl in sig_map.items():
        cl = [r0 for r0 in cl if r0 not in used]
        if len(cl) >= 2:
            clusters.append((cl, [p for p, _ in sig],
                             [rho for _, rho in sig]))
            used.update(cl)
    return clusters


def orbit_clusters(items):
    """items: [(key, sig)]. Returns [([keys], sig)]: disjoint clusters of
    size >= 2 sharing one orbit signature (first-signature-wins in item
    order). Used identically by the ILP pricing and the emitter."""
    by_sig = {}
    for key, sig in items:
        if sig:
            by_sig.setdefault(sig, []).append(key)
    clusters = []
    used = set()
    for sig, ks in by_sig.items():
        ks = [k for k in ks if k not in used]
        if len(ks) >= 2:
            clusters.append((ks, sig))
            used.update(ks)
    return clusters


def solve_problem(problem, ext, time_limit=300, sums=True, group=True,
                  partial_keep=True):
    # reps with non-identity perm members may not join partial-keep families
    # (the S-sum bookkeeping across shifted keep sets is unsound there)
    """One joint ILP over ALL outputs: cross-output sharing via a shared
    universe; term permutation orbits collapsed (representative solved once,
    members ride as free permutes). Returns (obj, stats)."""
    all_terms = []
    owner = []
    per_out = []
    for ov in problem["reference"]["outputs"]:
        terms, r = parse_terms(problem, ov)
        for t in terms:
            t["out_rank"] = r
        if group:
            groups, perms, tsigns = term_groups(terms)
            reps = sorted(m[0] for _, m in groups)
            nmemb = {m[0]: len(m) for _, m in groups}
            for t in reps:
                terms[t]["nmemb"] = nmemb[t]
            grouped_terms = [terms[t] for t in reps]
        else:
            groups = [(None, [t]) for t in range(len(terms))]
            perms = {t: list(range(r)) for t in range(len(terms))}
            tsigns = {t: (canon_full(terms[t]["factors"], terms[t]["dom"],
                                     set(range(r)))[2] if SYMS else 1)
                      for t in range(len(terms))}
            grouped_terms = terms
        base = len(all_terms)
        all_terms.extend(grouped_terms)
        owner.extend([ov] * len(grouped_terms))
        per_out.append({"ov": ov, "terms": terms, "rank": r,
                        "groups": groups, "perms": perms, "tsigns": tsigns,
                        "rep_base": base})
    rep_meta = {}
    rep_pout = {}
    rep_orbit = {}
    for pod in per_out:
        base = pod["rep_base"]
        p_out = 1
        for p in range(pod["rank"]):
            p_out *= ext[pod["terms"][0]["dom"][p]]
        for k, (_, members) in enumerate(pod["groups"]):
            rep = members[0]
            rep_meta[base + k] = {
                "members": [(t, pod["perms"][t]) for t in members],
                "rank": pod["rank"],
            }
            rep_pout[base + k] = (pod["ov"], p_out)
            rep_orbit[base + k] = orbit_signature(
                pod["terms"], members, pod["perms"], pod["tsigns"])
    res = solve_global(all_terms, ext, time_limit, sums,
                       partial_keep=partial_keep, rep_meta=rep_meta,
                       rep_pout=rep_pout, rep_orbit=rep_orbit)
    if res[0] is None:
        return None
    (obj, sol, U, stats), _ = res
    return {"obj": obj, "sol": sol, "U": U, "all_terms": all_terms,
            "per_out": per_out, "stats": stats}\
        if True else None


def solve_global(terms, ext, time_limit=300, sums=True,
                 partial_keep=True, rep_meta=None, rep_pout=None,
                 rep_orbit=None):
    """Exact multi-level sharing ILP for one definition, plus bounded
    sum-intermediate candidates (factor-out groups over rest families).
    Returns (obj, stats) — cost only, no emission."""
    t0 = time_mod.time()
    U, term_blocks, term_entries = build_universe(terms)

    # use-context count: terms containing the sig + sigs containing it as a
    # proper sub-block
    nctx = defaultdict(int)
    for t, blocks in enumerate(term_blocks):
        for sig in {s for _, s in blocks}:
            nctx[sig] += 1
    sub_of = {}
    for sig, rec in U.items():
        subs = []
        for idxs, subsig, subfree, subren, subsgn in subset_sigs(
                rec["cfactors"], rec["cdom"], rec["cfree"]):
            subs.append((idxs, subsig, subfree, subren, subsgn))
        sub_of[sig] = subs
        for _, ss, _, _, _ in subs:
            nctx[ss] += 1

    cand = [s for s in U if nctx[s] >= 2]
    candset = set(cand)

    # formations per candidate sig
    forms = {}   # sig -> [(cover, cost)]  cover = tuple of (idxs, subsig)
    for s in cand:
        rec = U[s]
        blocks = [(idxs, ss, free_in_m, ren_in_m, sgn_in_m) for
                  idxs, ss, free_in_m, ren_in_m, sgn_in_m in sub_of[s]
                  if ss in candset]
        atom_sets = [set(ls) for _, ls in rec["cfactors"]]
        forms[s] = cover_choices(atom_sets, blocks, rec["cfree"],
                                 rec["cdom"], ext)

    # term-level choices
    tchoices = []
    for t, term in enumerate(terms):
        blocks = [(idxs, sig, free, ren, sgn) for
                  idxs, sig, free, ren, sgn in term_entries[t]
                  if sig in candset]
        atom_sets = [set(ls) for _, ls in term["factors"]]
        tchoices.append(cover_choices(atom_sets, blocks,
                                      list(range(term["rank"])),
                                      term["dom"], ext))

    # ---- sum-intermediate candidates, cross-cofactor sharing:
    # families gamma=(gsig, sigma) with sigma=(cf_rest, out_g); sums S_sigma
    # are shared ACROSS families: s_sigma pure-linkage vars, h_gamma combine
    # vars, r_{gamma,t} per-member rest+add vars.
    # S-sum/combine and final-lincomb costs are PER OUTPUT: a family
    # spanning outputs is emitted once per owning output (B-007).
    rep_owner = {tt: rep_pout[tt][0] for tt in rep_pout} if rep_pout \
        else {}
    fam_ctx_raw = (sum_group_candidates(terms, term_entries,
                                        partial_keep=partial_keep)
                   if sums else {})
    # split each family's context into orbit-signature classes: a class is
    # internally orbit-consistent (C-016), so splitting never loses the
    # orbit ride; members of other classes keep their own variants
    fam_ctx = {}
    for key, ctx in fam_ctx_raw.items():
        if not rep_orbit:
            fam_ctx[key] = ctx
            continue
        by_sign = defaultdict(list)
        for e in ctx:
            by_sign[rep_orbit[e[0]][1]].append(e)
        if len(by_sign) == 1:
            fam_ctx[key] = ctx
        else:
            for ci2, (_sig, es) in enumerate(
                    sorted(by_sign.items(), key=lambda kv: str(kv[0]))):
                if len(es) >= 2:
                    fam_ctx[key + (ci2,)] = es

    # C-016: a family is orbit-consistent if all its covered reps share
    # one orbit signature; then the family result is emitted once and the
    # whole perm orbit rides it via free permutes (no standalone member
    # evals, no S-space bookkeeping for perm members).
    fam_sig = {}
    for gamma, ctx in fam_ctx.items():
        if rep_orbit:
            sigs = {rep_orbit[tt][1] for tt, _, _, _ in ctx}
            if len(sigs) == 1:
                fam_sig[gamma] = next(iter(sigs))
                continue
        fam_sig[gamma] = None

    fams = {}        # gamma -> [t]
    fam_g = {}       # gamma -> gsig
    fam_sigma = {}   # gamma -> sigma
    if sums:
        for key, ctx in fam_ctx.items():
            gsig, cf_rest, out_g = key[:3]
            fams[key] = [t for t, _, _, _ in ctx]
            fam_g[key] = gsig
            fam_sigma[key] = (cf_rest, out_g)
    sigmas = sorted({fam_sigma[g] for g in fams})

    # precompute per-(family, member) rest cost and per-sigma add cost
    rcost_rt = {}    # (gamma, t) -> dp(rest) + one add
    comb_cost = {}   # gamma -> combine + S-add + standalone-member cost

    # rests may themselves use materialized product intermediates: each
    # (family, member) gets rest-cover choices, exactly like terms
    rchoices = {}   # (gamma, t) -> [(cover, cost)]
    for gamma, ctx in fam_ctx.items():
        gsig, cf_rest, out_g = gamma[:3]
        _, cfree, cdom = canon_record(gsig)
        for (tt, idxs, _, _) in ctx:
            term = terms[tt]
            k = len(term["factors"])
            rest = [term["factors"][i] for i in range(k) if i not in idxs]
            rest_labels = {l for _, ls in rest for l in ls}
            keep = sorted({l for _, ls in
                           [term["factors"][i] for i in sorted(idxs)]
                           for l in ls} & rest_labels) + \
                sorted(p for p in range(term["rank"]) if p in rest_labels)
            # eligible blocks: candidate sigs that are subsets of the rest
            rblocks = []
            for size in range(2, len(rest) + 1):
                for cidx in combinations(range(len(rest)), size):
                    cidx = frozenset(cidx)
                    facs = [rest[i] for i in cidx]
                    labs = {l for _, ls in facs for l in ls}
                    outside = set(keep)
                    for j, (_, ls) in enumerate(rest):
                        if j not in cidx:
                            outside |= set(ls)
                    free = labs & outside
                    ssig, sren, ssgn = canon_full(facs, term["dom"], free)
                    if ssig in candset:
                        rblocks.append((cidx, ssig, free, sren, ssgn))
            atom_sets = [set(ls) for _, ls in rest]
            rchoices[(gamma, tt)] = cover_choices(atom_sets, rblocks, keep,
                                                  term["dom"], ext)
        # per-output S add + combine costs; only identity-perm members
        # ride in the S-sum (non-identity are standalone, B-006 fix 2)
        comb_cost[gamma] = 0
        by_out = {}
        for tt, _, _, _ in ctx:
            by_out.setdefault(rep_owner.get(tt), []).append(tt)
        for ov, tts in by_out.items():
            term0 = terms[tts[0]]
            p_add = 1     # S-space: cf_rest + keep positions
            for cid in cf_rest:
                p_add *= ext[cdom[cid]]
            pcomb = 1     # combine: cfree + keep positions
            for cid in cfree:
                pcomb *= ext[cdom[cid]]
            for p in range(term0["rank"]):
                if p not in out_g:
                    p_add *= ext[term0["dom"][p]]
                    pcomb *= ext[term0["dom"][p]]
            n_memb = 0
            for tt in tts:
                n_memb += 1
                meta = (rep_meta or {}).get(tt)
                if meta:
                    n_memb += sum(
                        1 for _lt, pi in meta["members"][1:]
                        if all(p == q for p, q in enumerate(pi)))
            comb_cost[gamma] += ((2 if cf_rest else 1) * pcomb
                                 + max(0, n_memb - 1) * p_add)
        # non-identity perm members of covered reps are emitted
        # separately (plain full evaluation) UNLESS the family is
        # orbit-consistent (C-016: they ride the family result for free)
        extra = 0
        if fam_sig.get(gamma) is None:
            for tt, _, _, _ in ctx:
                meta = (rep_meta or {}).get(tt)
                if not meta:
                    continue
                for _lt, pi in meta["members"][1:]:
                    if not all(p == q for p, q in enumerate(pi)):
                        term = terms[tt]
                        cst, _, _ = dp_tree([list(ls) for _, ls in
                                             term["factors"]],
                                            list(range(term["rank"])),
                                            term["dom"], ext)
                        extra += cst
        comb_cost[gamma] += extra

    # C-012/C-016 sum-orbit clusters over final-lincomb items: ungrouped
    # reps (z-covered) and orbit-consistent family results, clustered per
    # output by identical orbit signature. A cluster emits one pre-sum
    # that the whole orbit rides, saving (k-1)*m adds of the output size.
    clusters = []   # (items, saving); items = ("z", rep) | ("h", gamma)
    if rep_orbit and rep_pout:
        p_out_of = {ov: po for tt, (ov, po) in rep_pout.items()}
        items_by_out = defaultdict(list)
        for t in range(len(terms)):
            sig = rep_orbit.get(t, (None, None))[1]
            if sig:
                items_by_out[rep_pout[t][0]].append(("z", t, sig))
        for gamma, ctx in fam_ctx.items():
            sig = fam_sig.get(gamma)
            if sig:
                ovs = []
                for tt, _, _, _ in ctx:
                    ov = rep_pout[tt][0]
                    if ov not in ovs:
                        ovs.append(ov)
                for ov in ovs:
                    items_by_out[ov].append(("h", gamma, sig))
        for ov, items in items_by_out.items():
            for ks, sig in orbit_clusters(
                    [((it[0], it[1]), it[2]) for it in items]):
                saving = (len(ks) - 1) * len(sig) * p_out_of[ov]
                clusters.append((ks, saving))

    # ---- ILP
    x_idx = {s: i for i, s in enumerate(cand)}
    yvars = []   # (sig, cover, cost)
    for s in cand:
        for cover, cst in forms[s]:
            yvars.append((s, cover, cst))
    zvars = []   # (t, cover, cost)
    for t in range(len(terms)):
        for cover, cst in tchoices[t]:
            zvars.append((t, cover, cst))
    hvars = sorted(fam_ctx)                    # gamma
    svars = sorted({fam_sigma[g] for g in hvars})
    rvars = [(g, t, ci) for (g, t), chs in rchoices.items()
             for ci in range(len(chs))]

    nx, ny, nz = len(cand), len(yvars), len(zvars)
    nh, ns, nr = len(hvars), len(svars), len(rvars)
    nw = len(clusters)
    n = nx + ny + nz + nh + ns + nr + nw
    h0 = nx + ny + nz
    s0 = h0 + nh
    r0 = s0 + ns
    w0 = r0 + nr
    h_idx = {g: i for i, g in enumerate(hvars)}
    s_idx = {sg: i for i, sg in enumerate(svars)}
    r_idx = {rt: i for i, rt in enumerate(rvars)}
    r_cov = {rt: chs for rt, chs in rchoices.items()}

    c = np.zeros(n)
    for j, (_, _, cst) in enumerate(yvars):
        c[nx + j] = cst
    for j, (zt, _, cst) in enumerate(zvars):
        # final-lincomb add units: an ungrouped rep contributes 1 arg
        # (identity members fold into its coeff) plus one permuted arg
        # per non-identity member (B-007/C-012 exact accounting)
        n_nonid = 0
        meta = (rep_meta or {}).get(zt)
        if meta:
            n_nonid = sum(1 for _lt, pi in meta["members"][1:]
                          if not all(p == q for p, q in enumerate(pi)))
        c[nx + ny + j] = cst + (rep_pout[zt][1] * (1 + n_nonid)
                                if rep_pout else 0)
    for j, g in enumerate(hvars):
        # a chosen family contributes 1 + (#non-identity perm members of
        # its covered reps, emitted standalone) final-lincomb args
        lc_units = 0
        if rep_pout:
            # a family may span outputs (emitted once per owning output):
            # sum the per-output lincomb units. Orbit-consistent families
            # (C-016) emit one arg plus one permuted arg per orbit element;
            # others emit one standalone arg per non-identity member.
            fsig = fam_sig.get(g)
            ovs = []
            for tt, _, _, _ in fam_ctx[g]:
                ov, po = rep_pout[tt]
                if ov not in ovs:
                    ovs.append(ov)
            for ov in ovs:
                po = rep_pout[[tt for tt, _, _, _ in fam_ctx[g]
                               if rep_pout[tt][0] == ov][0]][1]
                if fsig is not None:
                    n_args = 1 + len(fsig)
                else:
                    n_args = 1
                    for tt, _, _, _ in fam_ctx[g]:
                        if rep_pout[tt][0] != ov:
                            continue
                        meta = (rep_meta or {}).get(tt)
                        if meta:
                            n_args += sum(
                                1 for _lt, pi in meta["members"][1:]
                                if not all(p == q
                                           for p, q in enumerate(pi)))
                lc_units += po * n_args
        c[h0 + j] = comb_cost[g] + lc_units
    for j, (g, t, ci) in enumerate(rvars):
        c[r0 + j] = r_cov[(g, t)][ci][1]
    # C-012/C-016 sum-orbit clusters: if every item of a cluster is
    # chosen (reps z-covered, families chosen), the final lincomb merges
    # the orbit over one pre-sum, saving (k-1)*m adds of the output size.
    for ci, (_items, saving) in enumerate(clusters):
        c[w0 + ci] = -saving

    rows, lb, ub = [], [], []
    # exactly one formation per materialized sig
    for i, s in enumerate(cand):
        row = lil_matrix((1, n))
        row[0, i] = 1.0
        for j, (ys, _, _) in enumerate(yvars):
            if ys == s:
                row[0, nx + j] = -1.0
        rows.append(row); lb.append(0.0); ub.append(0.0)
    for j, (ys, cover, _) in enumerate(yvars):
        for b in cover:
            row = lil_matrix((1, n))
            row[0, nx + j] = 1.0
            row[0, x_idx[b[1]]] = -1.0
            rows.append(row); lb.append(-np.inf); ub.append(0.0)
    # term coverage: plain cover OR one family membership
    for t in range(len(terms)):
        row = lil_matrix((1, n))
        for j, (zt, _, _) in enumerate(zvars):
            if zt == t:
                row[0, nx + ny + j] = 1.0
        for j, (g, tt, ci) in enumerate(rvars):
            if tt == t:
                row[0, r0 + j] = 1.0
        rows.append(row); lb.append(1.0); ub.append(1.0)
    for j, (zt, cover, _) in enumerate(zvars):
        for b in cover:
            row = lil_matrix((1, n))
            row[0, nx + ny + j] = 1.0
            row[0, x_idx[b[1]]] = -1.0
            rows.append(row); lb.append(-np.inf); ub.append(0.0)
    # r <= h, r <= x_B for blocks in the rest cover; h <= s; h <= x_g
    for j, (g, t, ci) in enumerate(rvars):
        row = lil_matrix((1, n))
        row[0, r0 + j] = 1.0
        row[0, h0 + h_idx[g]] = -1.0
        rows.append(row); lb.append(-np.inf); ub.append(0.0)
        for b in r_cov[(g, t)][ci][0]:
            row = lil_matrix((1, n))
            row[0, r0 + j] = 1.0
            row[0, x_idx[b[1]]] = -1.0
            rows.append(row); lb.append(-np.inf); ub.append(0.0)
    for j, g in enumerate(hvars):
        row = lil_matrix((1, n))
        row[0, h0 + j] = 1.0
        row[0, s0 + s_idx[fam_sigma[g]]] = -1.0
        rows.append(row); lb.append(-np.inf); ub.append(0.0)
        if g[0] in candset:
            row = lil_matrix((1, n))
            row[0, h0 + j] = 1.0
            row[0, x_idx[g[0]]] = -1.0
            rows.append(row); lb.append(-np.inf); ub.append(0.0)

    # w_C <= (item chosen): z-sum per rep, h per family
    for ci, (cl_items, _saving) in enumerate(clusters):
        for kind, idx in cl_items:
            row = lil_matrix((1, n))
            row[0, w0 + ci] = 1.0
            if kind == "z":
                for j, (zt, _, _) in enumerate(zvars):
                    if zt == idx:
                        row[0, nx + ny + j] = -1.0
            else:
                row[0, h0 + h_idx[idx]] = -1.0
            rows.append(row); lb.append(-np.inf); ub.append(0.0)

    from scipy.sparse import vstack
    A = vstack(rows).tocsr()
    t_build = time_mod.time() - t0
    res = milp(c=c, constraints=LinearConstraint(A, lb, ub),
               integrality=np.ones(n), bounds=Bounds(np.zeros(n), np.ones(n)),
               options={"time_limit": time_limit})
    stats = {"U": len(U), "candidates": len(cand), "vars": n,
             "constraints": A.shape[0], "build_s": t_build,
             "families": nh, "sigmas": ns, "status": res.status,
             "mip_gap": getattr(res, "mip_gap", None)}
    if res.x is None:
        return None, stats
    xs = res.x
    sol = {
        "fam_ctx": fam_ctx,
        "fam_sig": fam_sig,
        "x": {s for i, s in enumerate(cand) if xs[i] > 0.5},
        "y": {ys: cover for j, (ys, cover, _) in enumerate(yvars)
              if xs[nx + j] > 0.5},
        "z": {t: cover for j, (t, cover, _) in enumerate(zvars)
              if xs[nx + ny + j] > 0.5},
        "h": {hvars[j] for j in range(nh) if xs[h0 + j] > 0.5},
        "r": {rvars[j]: r_cov[(rvars[j][0], rvars[j][1])][rvars[j][2]][0]
              for j in range(nr) if xs[r0 + j] > 0.5},
    }
    return (res.fun, sol, U, stats), stats


# ------------------------------------------------------------------ the ILP

def keep_out_full(term):
    return tuple(range(term["rank"]))


def solve_definition(terms, ext, cap=None):
    """One definition (one output). Returns None or
    (obj, chosen, siginfo, groups, perms); chosen[rep_t] =
    (subset, grouped_sig_or_None), keyed by representative term index."""
    groups, perms, _ = term_groups(terms)
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
            # grouped variants: one ENTRY of the subset factors out; the rest
            # (including other entries, even of the same sig) sums first.
            for ei, e in enumerate(subset):
                g = e[1]
                rest = [sorted(x[2]) for j, x in enumerate(subset)
                        if j != ei] + singles
                if not rest:
                    continue
                rest_labels = set().union(*(set(a) for a in rest))
                # v1: group only full-keep rests (carry free(g) union out),
                # so the sum space is uniform and permutes stay pure.
                if not (set(e[2]) | set(range(term["rank"]))) <= rest_labels:
                    continue
                keep = sorted(e[2]) + list(range(term["rank"]))
                cst, _, _ = dp_tree(rest, keep, term["dom"], ext)
                add = 1
                for l in keep:
                    add *= ext[term["dom"][l]]
                ckeep = tuple(sorted(e[3][l] for l in e[2]))
                zvars.append((t, subset, (g, ckeep, keep_out_full(term), ei),
                              cst + add - osize))

    nz = len(zvars)
    xsigs = sigs
    nx = len(xsigs)
    # group vars keyed by (sig, canonical keep ids, keep-out, ) -- the
    # entry index within the subset is per-term and not part of the key
    gkeys = sorted({z[2][:3] for z in zvars if z[2] is not None})
    w_idx = {k: i for i, k in enumerate(gkeys)}
    wbase = nz + nx
    nw = len(gkeys)
    n = nz + nx + nw

    c = np.zeros(n)
    for j, (_, _, _, cst) in enumerate(zvars):
        c[j] = zvars[j][3]
    for i, s in enumerate(xsigs):
        c[nz + i] = icost[s]
    r0 = terms[0]["rank"]
    osize = 1
    for l in range(r0):
        osize *= ext[terms[0]["dom"][l]]
    for k, (s, ckeep, keep_out) in enumerate(gkeys):
        info = siginfo[s]
        p = 1
        for l in info["cfree"]:
            p *= ext[info["cdom"][l]]
        for l in keep_out:
            p *= ext[terms[0]["dom"][l]]
        # group cost: one combine g x S + one lincomb arg; the intermediate
        # itself is charged via x_s (constraint w <= x below)
        c[wbase + k] = 2 * p + osize
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
        for ei, e in enumerate(subset):
            row = lil_matrix((1, n))
            row[0, j] = 1.0
            if grouped is not None and ei == grouped[3]:
                row[0, wbase + w_idx[grouped[:3]]] = -1.0  # needs w_group
            else:
                row[0, nz + x_idx[e[1]]] = -1.0            # needs x_g
            rows.append(row)
            lb.append(-np.inf)
            ub.append(0.0)
    # group requires its intermediate materialized: w_gk <= x_sig
    for k, (s, _, _) in enumerate(gkeys):
        row = lil_matrix((1, n))
        row[0, wbase + k] = 1.0
        row[0, nz + x_idx[s]] = -1.0
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

def emit_binary_tree(values, leaf_info, out_order, dom, ext, sign=1):
    """Emit a binary contraction tree over `leaf_info` = [(argref_dict,
    labels tuple in that arg's own axis order)]. The root's declared output
    order is `out_order`; internal nodes keep labels needed later, sorted.
    `sign` folds a +/-1 into the root step's coeff (B-006 formation signs).
    Returns (argref of result, output labels tuple)."""
    if len(leaf_info) == 1 and sign == 1:
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

    if len(leaf_info) == 1:
        # sign == -1 with a single leaf: emit a scaling step
        ref, ls = leaf_info[0]
        vid = len(values)
        labs = sorted(set(ls) | set(out_order))
        values.append({"Einsum": {
            "coeff": [sign, 1],
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
            "coeff": [sign if is_root else 1, 1],
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
        members_of = {m[0]: m for _, m in groups}

        indiv, grouped = set(), defaultdict(list)
        for rep_t, (subset, g) in chosen.items():
            for ei, e in enumerate(subset):
                if g is not None and ei == g[3]:
                    grouped[g[:3]].append(rep_t)
                else:
                    indiv.add(e[1])

        sig_value = {}
        for s in sorted(indiv | {g for (g, _, _) in grouped}):
            info = siginfo[s]
            leaf_info = [({"Tensor": tid}, ls) for tid, ls in
                         info["cfactors"]]
            ref, out = emit_binary_tree(values, leaf_info,
                                        list(info["cfree"]), info["cdom"],
                                        ext)
            sig_value[s] = (ref, out)

        def emit_rest(rep_t, keep_term_labels, skip_sig, skip_ei=-1):
            term = terms[rep_t]
            subset, _ = chosen[rep_t]
            usedf = frozenset().union(*(e[0] for e in subset)) if subset \
                else frozenset()
            leaf_info = []
            for ei, e in enumerate(subset):
                if ei == skip_ei:
                    continue
                _, sig, free, ren = e
                ref, order = sig_value[sig]
                ren_inv = {v: k for k, v in ren.items()}
                leaf_info.append((ref, tuple(ren_inv[c] for c in order)))
            for i, (tid, ls) in enumerate(term["factors"]):
                if i not in usedf:
                    leaf_info.append(({"Tensor": tid}, tuple(ls)))
            return emit_binary_tree(values, leaf_info, keep_term_labels,
                                    term["dom"], ext)

        contribs = []   # (argref, coeff) for the final lincomb

        for (g, ckeep, keep_out), reps in grouped.items():
            info = siginfo[g]
            cfree = list(info["cfree"])
            s_order = [S_BASE + c for c in ckeep] + list(keep_out)
            s_dom = {str(S_BASE + c): info["cdom"][c] for c in ckeep}
            for l in keep_out:
                s_dom[str(l)] = terms[0]["dom"][l]
            sum_args, sum_coeffs, sum_inputs = [], [], []
            for rep_t in reps:
                ge = chosen[rep_t][0][chosen[rep_t][1][3]]
                ren_inv = {v: k for k, v in ge[3].items()}
                keep_term = [ren_inv[c] for c in ckeep] + list(keep_out)
                ref, _ = emit_rest(rep_t, keep_term, g, chosen[rep_t][1][3])
                for t in members_of[rep_t]:
                    pi = perms[t]
                    if all(p == q for p, q in enumerate(pi)):
                        sum_args.append(ref)
                    else:
                        out_tuple = ([S_BASE + c for c in ckeep]
                                     + [pi[q] for q in keep_out])
                        used = set(s_order) | set(out_tuple)
                        vid = len(values)
                        values.append({"Einsum": {
                            "coeff": [1, 1],
                            "code": {
                                "inputs": [s_order],
                                "output": out_tuple,
                                "domains": {str(l): s_dom[str(l)]
                                            if str(l) in s_dom else
                                            terms[0]["dom"][l]
                                            for l in sorted(used)}},
                            "args": [ref]}})
                        sum_args.append({"Value": vid})
                    sum_coeffs.append(terms[t]["coeff"])
            vid = len(values)
            values.append({"LinComb": {
                "coeff": [[c.numerator, c.denominator] for c in sum_coeffs],
                "code": {"inputs": sum_inputs,
                         "output": s_order, "domains": dict(s_dom)},
                "args": sum_args}})
            # combine g x S -> out (rep space)
            rep0 = reps[0]
            ge0 = chosen[rep0][0][chosen[rep0][1][3]]
            ren_inv0 = {v: k for k, v in ge0[3].items()}
            gref, gorder = sig_value[g]
            gin = [S_BASE + cid if cid in ckeep else ren_inv0[cid]
                   for cid in gorder]
            step_dom = dict(s_dom)
            for l in gin + s_order:
                if l < S_BASE and str(l) not in step_dom:
                    step_dom[str(l)] = terms[0]["dom"][l]
            vid2 = len(values)
            values.append({"Einsum": {
                "coeff": [1, 1],
                "code": {"inputs": [gin, s_order],
                         "output": list(range(r)),
                         "domains": step_dom},
                "args": [gref, {"Value": vid}]}})
            contribs.append(({"Value": vid2}, Fraction(1)))

        # ungrouped reps: evaluate once, permute per member
        for rep_t, (subset, g) in chosen.items():
            if g is not None:
                continue
            ref, _ = emit_rest(rep_t, list(range(r)), None)
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
                                 "output": [pi_inv[p] for p in range(r)],
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


# ------------------------------------------------- global-arm emission

def member_instance_signs(mterm, n_cofactor, gsig, r):
    """(s_g, s_rest): orbit signs of the member's cofactor instance and its
    rest vs the shared canonical forms. +1/+1 when no symmetries."""
    if not SYMS:
        return 1, 1
    for cidx in combinations(range(len(mterm["factors"])), n_cofactor):
        facs = [mterm["factors"][i] for i in cidx]
        labs = {l for _, ls in facs for l in ls}
        outside = set(range(r))
        for j, (_, ls) in enumerate(mterm["factors"]):
            if j not in cidx:
                outside |= set(ls)
        free = labs & outside
        msig, mren, msgn = canon_full(facs, mterm["dom"], free)
        if msig == gsig:
            rest = [mterm["factors"][i]
                    for i in range(len(mterm["factors"])) if i not in cidx]
            rest_labs = {l for _, ls in rest for l in ls}
            s_rest = canon_full(rest, mterm["dom"], rest_labs)[2]
            return msgn, s_rest
    raise RuntimeError("member instance not found")


def emit_global(solpack, ext):
    """Emit one Program from a joint global-ILP solution.

    Sign conventions (verified by the oracle): every emitted intermediate is
    its canonical form; a written instance equals (orbit sign) x canonical,
    so true contributions pick up coeff x (product of instance signs).
    Permutes use output=[pi_inv(q)] (settled by numpy microtest)."""
    U = solpack["U"]
    sol = solpack["sol"]
    per_out = solpack["per_out"]
    all_terms = solpack["all_terms"]
    fam_ctx = sol["fam_ctx"]
    values = []

    # --- materialized intermediates, smallest first (blocks are strict
    #     subsets: factor count is a valid topological order)
    sig_ref = {}
    for s in sorted(sol["x"], key=lambda s: len(U[s]["cfactors"])):
        rec = U[s]
        cover = sol["y"][s]
        used = frozenset().union(*(b[0] for b in cover)) if cover \
            else frozenset()
        leaf_info = []
        fsign = 1   # product of block orbit signs (B-006): the block
        # instance inside this canonical form is fsign x canonical
        for (idxs, subsig, _free, ren, bsgn) in cover:
            fsign *= bsgn
            bref, border = sig_ref[subsig]
            ren_inv = {v: k for k, v in ren.items()}
            leaf_info.append((bref, tuple(ren_inv[c] for c in border)))
        for i, (tid, ls) in enumerate(rec["cfactors"]):
            if i not in used:
                leaf_info.append(({"Tensor": tid}, tuple(ls)))
        ref, out = emit_binary_tree(values, leaf_info, list(rec["cfree"]),
                                    rec["cdom"], ext, sign=fsign)
        sig_ref[s] = (ref, out)

    def cofactor_ref(gsig):
        if gsig in sig_ref:
            return sig_ref[gsig]
        body, cfree, cdom = canon_record(gsig)
        assert len(body) == 1
        tid, ls = body[0]
        return {"Tensor": tid}, tuple(ls)

    grouped_reps = {t for (_, t, _) in sol["r"]}

    outputs = []
    for pod in per_out:
        terms = pod["terms"]
        r = pod["rank"]
        perms = pod["perms"]
        tsigns = pod["tsigns"]
        groups = pod["groups"]
        members_of = {m[0]: m for _, m in groups}
        reps = sorted(m[0] for _, m in groups)
        base = pod["rep_base"]

        contribs = []
        fam_items = []   # (gamma, ref): orbit-consistent family results
        fam_sig = sol.get("fam_sig", {})

        for gamma in sorted(sol["h"]):
            members = [(t, ci) for (g2, t, ci) in sol["r"]
                       if g2 == gamma and base <= t < base + len(reps)]
            if not members:
                continue
            gsig, cf_rest, out_g = gamma[:3]
            _, g_cfree, g_cdom = canon_record(gsig)
            keep_out = [p for p in range(r) if p not in out_g]
            s_order = [S_BASE + c for c in cf_rest] + keep_out
            s_dom = {str(S_BASE + c): g_cdom[c] for c in cf_rest}
            for l in keep_out:
                s_dom[str(l)] = terms[0]["dom"][l]
            sum_args, sum_coeffs, sum_inputs = [], [], []
            first_ren_inv = None
            for (tg, ci) in members:
                local_rep = reps[tg - base]
                term = terms[local_rep]
                rentry = [e for e in fam_ctx[gamma] if e[0] == tg][0]
                ren = rentry[2]
                s_g = rentry[3]
                ren_inv = {v: k for k, v in ren.items()}
                if first_ren_inv is None:
                    first_ren_inv = ren_inv
                keep_term = [ren_inv[c] for c in cf_rest] + keep_out
                cover = sol["r"][(gamma, tg, ci)]
                # rep's rest multiplier = block signs; member multiplier
                # additionally needs canon-sign ratio of written rests
                rest = [term["factors"][i]
                        for i in range(len(term["factors"]))
                        if i not in rentry[1]]
                s_rest = 1
                for b in cover:
                    if len(b) > 4:
                        s_rest *= b[4]
                s_rest_rep_written = canon_full(rest, term["dom"],
                                                set(keep_term))[2] \
                    if SYMS else 1
                leaf_info = []
                covered_rest = set()
                for b in cover:
                    bref, border = sig_ref[b[1]]
                    bren_inv = {v: k for k, v in b[3].items()}
                    leaf_info.append((bref, tuple(bren_inv[c]
                                                  for c in border)))
                    covered_rest |= set(b[0])
                for i, (tid, ls) in enumerate(rest):
                    if i not in covered_rest:
                        leaf_info.append(({"Tensor": tid}, tuple(ls)))
                ref, _ = emit_binary_tree(values, leaf_info, keep_term,
                                          term["dom"], ext)
                # members: identity ride directly; non-identity members
                # are emitted separately as plain full evaluations (the
                # post-family section) and charged in the ILP — they must
                # NOT join the S-sum (B-006: the pi-mapped own-rest member
                # cancels the rep's rest under antisymmetry).
                for t in members_of[local_rep]:
                    pi = perms[t]
                    is_id = all(p == q for p, q in enumerate(pi))
                    if not is_id:
                        continue
                    sum_args.append(ref)
                    sum_inputs.append(list(s_order))
                    if t == local_rep:
                        sum_coeffs.append(terms[t]["coeff"]
                                          * s_g * s_rest)
                    else:
                        s_g_m, s_rest_m = member_instance_signs(
                            terms[t], len(rentry[1]), gsig, r)
                        sum_coeffs.append(terms[t]["coeff"]
                                          * s_g_m * s_rest_m
                                          * s_rest_rep_written * s_rest)
            vid = len(values)
            values.append({"LinComb": {
                "coeff": [[c.numerator, c.denominator] for c in sum_coeffs],
                "code": {"inputs": sum_inputs,
                         "output": s_order, "domains": dict(s_dom)},
                "args": sum_args}})
            gref, gorder = cofactor_ref(gsig)
            gin = [S_BASE + cid if cid in cf_rest else first_ren_inv[cid]
                   for cid in gorder]
            step_dom = dict(s_dom)
            for l in gin:
                if l < S_BASE and str(l) not in step_dom:
                    step_dom[str(l)] = terms[0]["dom"][l]
            vid2 = len(values)
            values.append({"Einsum": {
                "coeff": [1, 1],
                "code": {"inputs": [gin, s_order], "output": list(range(r)),
                         "domains": step_dom},
                "args": [gref, {"Value": vid}]}})
            if fam_sig.get(gamma):
                # C-016: orbit-consistent — the whole perm orbit rides
                # the family result; members are NOT emitted standalone
                fam_items.append((gamma, {"Value": vid2}))
            else:
                contribs.append(({"Value": vid2}, Fraction(1)))

        # non-identity members of grouped reps: own plain evaluations
        # (skipped for orbit-consistent families, whose orbit rides free)
        for (g2, tg, ci) in sorted(sol["r"]):
            if fam_sig.get(g2):
                continue
            if not (base <= tg < base + len(reps)):
                continue
            local_rep = reps[tg - base]
            for t in members_of[local_rep][1:]:
                pi = perms[t]
                if all(p == q for p, q in enumerate(pi)):
                    continue
                mterm = terms[t]
                leaf_info = [({"Tensor": tid}, tuple(ls))
                             for tid, ls in mterm["factors"]]
                ref, _ = emit_binary_tree(values, leaf_info,
                                          list(range(r)),
                                          mterm["dom"], ext)
                contribs.append((ref, mterm["coeff"]))  # plain eval: exact

        # ungrouped reps: evaluate once; perm members ride via permutes.
        # Identity-perm members fold into the rep's coeff (no extra arg).
        # C-012 sum-orbits: clusters of reps with identical member
        # (perm, ratio) signatures share one pre-sum; the orbit rides it.
        rep_ref = {}
        for k, local_rep in enumerate(reps):
            tg = base + k
            if tg in grouped_reps:
                continue
            term = terms[local_rep]
            cover = sol["z"][tg]
            used = frozenset().union(*(b[0] for b in cover)) if cover \
                else frozenset()
            leaf_info = []
            for b in cover:
                bref, border = sig_ref[b[1]]
                bren_inv = {v: k for k, v in b[3].items()}
                leaf_info.append((bref, tuple(bren_inv[c] for c in border)))
            for i, (tid, ls) in enumerate(term["factors"]):
                if i not in used:
                    leaf_info.append(({"Tensor": tid}, tuple(ls)))
            ref, _ = emit_binary_tree(values, leaf_info, list(range(r)),
                                      term["dom"], ext)
            block_sign = 1
            for b in cover:
                if len(b) > 4:
                    block_sign *= b[4]
            rep_ref[local_rep] = (ref, block_sign)

        def rep_parts(local_rep):
            """(ref, folded identity coeff, [(perm, member coeff)])."""
            ref, bs = rep_ref[local_rep]
            c_eff = Fraction(0)
            nonid = []
            for t in members_of[local_rep]:
                pi = perms[t]
                sgn = tsigns[t] * tsigns[local_rep] if t != local_rep else 1
                if all(p == q for p, q in enumerate(pi)):
                    c_eff += terms[t]["coeff"] * sgn
                else:
                    nonid.append((pi, terms[t]["coeff"] * sgn * bs))
            return ref, c_eff * bs, nonid

        # C-012/C-016: final-lincomb items = ungrouped rep evals +
        # orbit-consistent family results, clustered by orbit signature.
        # A fully-present cluster emits one pre-sum that the orbit rides.
        gpos = {r0: base + k for k, r0 in enumerate(reps)}
        items = []       # (key, ref, base_coeff, sig)
        for local_rep in sorted(rep_ref):
            ref, c_eff, _nonid = rep_parts(local_rep)
            _, sig = orbit_signature(terms, members_of[local_rep], perms,
                                     tsigns)
            items.append((("z", gpos[local_rep]), ref, c_eff, sig))
        for gamma, fref in fam_items:
            items.append((("h", gamma), fref, Fraction(1),
                          fam_sig[gamma]))
        # candidate-item universe for this output (pricing parity): all
        # reps with a signature + all consistent families touching it
        cand_items = []
        for local_rep in reps:
            _, sig = orbit_signature(terms, members_of[local_rep], perms,
                                     tsigns)
            cand_items.append((("z", gpos[local_rep]), sig))
        for gamma, ctx in fam_ctx.items():
            if fam_sig.get(gamma) and any(
                    base <= e[0] < base + len(reps) for e in ctx):
                cand_items.append((("h", gamma), fam_sig[gamma]))
        present = {k for k, _, _, _ in items}
        item_by_key = {k: (ref, bc, sig) for k, ref, bc, sig in items}
        clustered = set()
        for ks, sig in orbit_clusters(cand_items):
            if not all(k in present for k in ks):
                continue   # partly unchosen: no merge (pricing agrees)
            vid = len(values)
            values.append({"LinComb": {
                "coeff": [[item_by_key[k][1].numerator,
                           item_by_key[k][1].denominator] for k in ks],
                "code": {"inputs": [list(range(r))] * len(ks),
                         "output": list(range(r)),
                         "domains": {str(l): terms[0]["dom"][l]
                                     for l in range(r)}},
                "args": [item_by_key[k][0] for k in ks]}})
            contribs.append(({"Value": vid}, Fraction(1)))
            for pi, rho in sig:
                pi_inv = [pi.index(q) for q in range(r)]
                outdom = {str(l): terms[0]["dom"][l] for l in range(r)}
                vid2 = len(values)
                values.append({"Einsum": {
                    "coeff": [1, 1],
                    "code": {"inputs": [list(range(r))],
                             "output": [pi_inv[p] for p in range(r)],
                             "domains": outdom},
                    "args": [{"Value": vid}]}})
                contribs.append(({"Value": vid2}, rho))
            clustered.update(ks)

        for key, ref, base_cf, sig in items:
            if key in clustered:
                continue
            if base_cf:
                contribs.append((ref, base_cf))
            if sig is None and key[0] == "z":
                # degenerate orbit signature: emit members directly
                _ref, _ce, nonid = rep_parts(
                    [r0 for r0 in rep_ref if gpos[r0] == key[1]][0])
                for pi, cf in nonid:
                    pi_inv = [pi.index(q) for q in range(r)]
                    outdom = {str(l): terms[0]["dom"][l] for l in range(r)}
                    vid = len(values)
                    values.append({"Einsum": {
                        "coeff": [1, 1],
                        "code": {"inputs": [list(range(r))],
                                 "output": [pi_inv[p] for p in range(r)],
                                 "domains": outdom},
                        "args": [ref]}})
                    contribs.append(({"Value": vid}, cf))
                continue
            for pi, rho in (sig or ()):
                pi_inv = [pi.index(q) for q in range(r)]
                outdom = {str(l): terms[0]["dom"][l] for l in range(r)}
                vid = len(values)
                values.append({"Einsum": {
                    "coeff": [1, 1],
                    "code": {"inputs": [list(range(r))],
                             "output": [pi_inv[p] for p in range(r)],
                             "domains": outdom},
                    "args": [ref]}})
                contribs.append(({"Value": vid}, rho * base_cf))

        outdom = {str(l): terms[0]["dom"][l] for l in range(r)}
        vid = len(values)
        values.append({"LinComb": {
            "coeff": [[cf.numerator, cf.denominator] for _, cf in contribs],
            "code": {"inputs": [list(range(r))] * len(contribs),
                     "output": list(range(r)),
                     "domains": outdom},
            "args": [ref for ref, _ in contribs]}})
        outputs.append(vid)
    return {"values": values, "outputs": outputs}


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
    global SYMS
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
    elif cmd == "solveg":
        prob_path, sizes = sys.argv[2], parse_sizes(sys.argv[3])
        problem = json.load(open(prob_path))
        if len(sys.argv) > 4 and sys.argv[4] == "sym":
            SYMS = problem["symmetries"]
        t_start = time_mod.time()
        out = solve_problem(problem, sizes)
        print(f"{prob_path}: joint global-ILP {int(out['obj'])} "
              f"(log10 {np.log10(out['obj']):.6f}) | "
              f"vars={out['stats']['vars']} cons={out['stats']['constraints']} "
              f"build={out['stats']['build_s']:.1f}s "
              f"status={out['stats']['status']} | "
              f"total wall {time_mod.time()-t_start:.1f}s")
    elif cmd == "solvegemit":
        prob_path, sizes, out_path = sys.argv[2], parse_sizes(sys.argv[3]), \
            sys.argv[4]
        problem = json.load(open(prob_path))
        if len(sys.argv) > 5 and sys.argv[5] == "sym":
            SYMS = problem["symmetries"]
        flags = set(sys.argv[5:])
        pack = solve_problem(problem, sizes, partial_keep=True,
                             sums="nosums" not in flags,
                             group="nogroup" not in flags)
        prog = emit_global(pack, sizes)
        priced = price_candidate(prog, sizes)
        json.dump(prog, open(out_path, "w"))
        print(f"emitted {len(prog['values'])} steps -> {out_path}")
        print(f"ILP objective {int(pack['obj'])} | emitted priced {priced}")


    else:
        print(__doc__)
        sys.exit(1)


def parse_sizes(s):
    return {int(k): int(v) for k, v in
            (kv.split("=") for kv in s.split(","))}


if __name__ == "__main__":
    main()
