#!/usr/bin/env python3
"""v3 wall-time regime pricer (joint-search arm).

Faithful Python port of the mainline cost model v3
(crates/harness/src/costmodel.rs @ 473e094, F-004/F-016 regime rules),
including the deterministic contraction pairing (det_greedy_pairs, F-028)
and tenferro's per-step GEMM plan classification (classify_modes /
compile_pairwise_step_plan semantics, tenferro-einsum 0.7.1).

Rates are DATA-DRIVEN from profiles/calibrate-v2.json (the mainline's M1
deliverable): each field names the calibration case(s) it reads. Never
hardcode a rate here. Structural constants (cache cliffs, floors) are
ported from costmodel.rs with their calibration provenance.

Layout caveat (task-specified): candidates carry logical label orders; the
harness applies `--layout model` post-emission. This pricer prices the
as-written orders; the divergence on known candidates is measured in
trials before the numbers are trusted (see NOTES.md).
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CALIB = os.path.join(ROOT, "profiles", "calibrate-v2.json")

# Structural constants ported from costmodel.rs @ 473e094 (calibration
# provenance: L3-effective knee measured between 21 and 43 MB of traffic on
# i5-13400F, physical L3 20 MB; slow-GEMM cliff at ~100 MB by deep k-reuse;
# floors from the small-scale grid records).
L3_BYTES = 40e6
SLOW_CLIFF_BYTES = 100e6
STEP_FLOOR_HI = 1e-4       # materializing step, out rank >= 3
STEP_FLOOR_OUT2 = 1e-4     # materializing step, out rank == 2
LINCOMB_FLOOR_HI = 8e-5    # per rank>=5 lincomb input (permute_r6 small)
DISPATCH = 15e-6


def _load_records(path=None):
    doc = json.load(open(path or CALIB))
    return {(r["case"], r["o"], r["v"], r["threads"]): r
            for r in doc["records"]}


class Rates:
    """Per-thread-count rates, derived from the calibration records."""

    # field -> (case, o, v), reading its gflops / gbytes_s
    GFLOPS = {
        "r_gemm": ("gemm_clean", 0, 0),
        "r_slow": ("ccd_slowpath_anchor", 16, 64),
        "r_slow_in": ("ccd_slowpath_incache", 12, 48),
        "r_reduce": ("bin_r6_scalar", 12, 24),
        "r_reduce_out1": ("bin_r6_out1", 12, 24),
    }
    GBYTES = {
        "bw_lo_in": ("kheavy_r2out", 16, 32),
        "bw_lo_out": ("kheavy_r2out", 24, 48),
        "bw_hi_in": ("permute_r6", 4, 8),
        "bw_hi_out": ("permute_r6", 12, 24),
        "bw_mul": ("tern_r6_chain_step1_mul", 8, 16),
        "bw_mul_out": ("tern_r6_chain_step1_mul", 12, 24),
        "bw_view": ("permute_r4", 16, 64),
    }

    def __init__(self, threads, path=None):
        t = 16 if threads >= 8 else 1
        recs = _load_records(path)
        for field, (case, o, v) in self.GFLOPS.items():
            setattr(self, field, recs[(case, o, v, t)]["gflops"] * 1e9)
        for field, (case, o, v) in self.GBYTES.items():
            setattr(self, field, recs[(case, o, v, t)]["gbytes_s"] * 1e9)
        self.l3_bytes = L3_BYTES
        self.slow_cliff_bytes = SLOW_CLIFF_BYTES
        self.step_floor_hi = STEP_FLOOR_HI
        self.step_floor_out2 = STEP_FLOOR_OUT2
        self.lincomb_floor_hi = LINCOMB_FLOOR_HI
        self.dispatch = DISPATCH
        self.threads = t


# ----------------------------------------------------------- label helpers

def _sizes_of(labels, dom, ext):
    return [ext[dom[l]] for l in labels]


def _prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


def _bytes(labels, dom, ext):
    return float(_prod(_sizes_of(labels, dom, ext))) * 8.0


# ------------------------------------------- det_greedy_pairs (F-028 port)

def det_greedy_pairs(inputs, output, size_of):
    """Faithful port of costmodel.rs::det_greedy_pairs (itself a port of
    tenferro-einsum 0.7.1 planning/tree.rs::optimize_self_greedy_pairs):
    repeatedly contract the pair whose intermediate is smallest; ties to
    the lexicographically first pair. Set-membership label counts,
    first-appearance intermediate order."""
    n = len(inputs)
    if n <= 1:
        return []
    available = list(range(n))
    operand_subs = [list(s) for s in inputs]
    needed = {}
    for l in output:
        needed[l] = needed.get(l, 0) + 1
    for subs in operand_subs:
        for l in set(subs):
            needed[l] = needed.get(l, 0) + 1

    def intermediate(left, right):
        out = []
        for l in list(left) + list(right):
            selected = (l in left) + (l in right)
            if needed.get(l, 0) > selected and l not in out:
                out.append(l)
        return out

    pairs = []
    while len(available) > 1:
        best = None
        for i in range(len(available)):
            for j in range(i + 1, len(available)):
                li, lj = available[i], available[j]
                subs = intermediate(operand_subs[li], operand_subs[lj])
                cost = max(1, _prod([size_of.get(l, 1) for l in subs]))
                if best is None or cost < best[0]:
                    best = (cost, i, j)
        _, bi, bj = best
        left, right = available[bi], available[bj]
        pairs.append((left, right))
        new_subs = intermediate(operand_subs[left], operand_subs[right])
        for subs in (operand_subs[left], operand_subs[right]):
            for l in set(subs):
                needed[l] -= 1
                if needed[l] == 0:
                    del needed[l]
        for l in set(new_subs):
            needed[l] = needed.get(l, 0) + 1
        new_idx = len(operand_subs)
        operand_subs.append(new_subs)
        available.remove(right)
        available.remove(left)
        available.append(new_idx)
    return pairs


# --------------------------------------- step planning (tenferro plan port)

def _intermediate_subs(subs_left, subs_right, needed):
    """Port of tenferro util.rs::intermediate_subs (first-appearance order,
    membership by set)."""
    out, seen = [], set()
    for l in list(subs_left) + list(subs_right):
        if l in needed and l not in seen:
            seen.add(l)
            out.append(l)
    return out


def _classify_modes(a, b, c):
    r"""Port of classify.rs::classify_modes: batch (A∩B∩C, A-order),
    lo (A∩C\B, A-order), ro (B∩C\A, B-order), sum (A∩B\C, A-order)."""
    set_a, set_b, set_c = set(a), set(b), set(c)
    batch, lo, ro, sum_ = [], [], [], []
    seen = set()
    for m in a:
        if m in seen:
            continue
        seen.add(m)
        if m in set_b and m in set_c:
            batch.append(m)
        elif m in set_c:
            lo.append(m)
        elif m in set_b:
            sum_.append(m)
    for m in b:
        if m in set_c and m not in set_a and m not in ro:
            ro.append(m)
    return batch, lo, ro, sum_


class StepPlan:
    __slots__ = ("reduce_a", "reduce_b", "m", "n", "k", "batch",
                 "target_a", "target_b", "canonical_modes", "lhs", "rhs",
                 "out")


def _step_plan(lhs, rhs, out, dom, ext):
    """Port of compile_pairwise_step_plan (plan.rs), minus diag stages
    (repeated labels inside one operand are not produced by this arm's
    emission; the harness would plan them, flagged if they appear)."""
    sp = StepPlan()
    sp.lhs, sp.rhs, sp.out = lhs, rhs, out
    set_b, set_c = set(rhs), set(out)
    kept_a = [l for l in lhs if l in set_b or l in set_c]
    sp.reduce_a = len(kept_a) != len(lhs)
    set_a = set(lhs)
    kept_b = [l for l in rhs if l in set_a or l in set_c]
    sp.reduce_b = len(kept_b) != len(rhs)
    eff_a, eff_b = kept_a, kept_b
    batch, lo, ro, sum_ = _classify_modes(eff_a, eff_b, out)
    sp.m = _prod(_sizes_of(lo, dom, ext))
    sp.n = _prod(_sizes_of(ro, dom, ext))
    sp.k = _prod(_sizes_of(sum_, dom, ext))
    sp.batch = _prod(_sizes_of(batch, dom, ext))
    sp.target_a = lo + sum_ + batch
    sp.target_b = sum_ + ro + batch
    sp.canonical_modes = lo + ro + batch
    return sp


def _tree_steps(inputs, output, size_of):
    """Simulate ContractionTree::from_pairs: operand subscripts evolve per
    det_greedy_pairs; yields (lhs_subs, rhs_subs, out_subs) per step."""
    pairs = det_greedy_pairs(inputs, output, size_of)
    operand_subs = [list(s) for s in inputs]
    live = [True] * len(inputs)
    steps = []
    for step_idx, (left, right) in enumerate(pairs):
        needed = set(output)
        for idx, subs in enumerate(operand_subs):
            if idx != left and idx != right and live[idx]:
                needed.update(subs)
        new_subs = _intermediate_subs(operand_subs[left],
                                      operand_subs[right], needed)
        is_last = step_idx == len(pairs) - 1
        steps.append((operand_subs[left], operand_subs[right],
                      output if is_last else new_subs))
        operand_subs.append(new_subs)
        live.append(True)
        live[left] = live[right] = False
    return steps


# ------------------------------------------------------------- the pricer

_LAYOUT_MODE = "clean_assumed"  # or "as_written" (divergence reporting)


def set_layout_mode(mode):
    global _LAYOUT_MODE
    assert mode in ("clean_assumed", "as_written")
    _LAYOUT_MODE = mode


def step_wall_sets(la, lb, out_set, dom, ext, rates):
    """v3 step price from label SETS under the clean-assumed layout:
    the emitter/harness layout pass is assumed to put operands at canonical
    target order (measured divergence vs harness --layout model pricing:
    totals 0.78-0.83x on adc2/ccsd_gristmill, i.e. this UNDER-prices by
    ~20%; ranking-oriented, recorded in NOTES.md). Reduce-pre-pass traffic
    and class floors are layout-independent and kept."""
    la, lb, out_set = set(la), set(lb), set(out_set)
    batch = la & lb & out_set
    lo = (la & out_set) - lb
    ro = (lb & out_set) - la
    sum_ = (la & lb) - out_set
    m = _prod(_sizes_of(sorted(lo), dom, ext))
    n = _prod(_sizes_of(sorted(ro), dom, ext))
    k = _prod(_sizes_of(sorted(sum_), dom, ext))
    bt = _prod(_sizes_of(sorted(batch), dom, ext))
    flops = 2.0 * m * n * k * bt
    out_rank = len(out_set)
    step_bytes = _bytes(sorted(la), dom, ext) + _bytes(sorted(lb), dom, ext) \
        + _bytes(sorted(out_set), dom, ext)
    max_rank = max(out_rank, len(la), len(lb))

    def bw_traffic():
        if max_rank >= 5:
            return rates.bw_hi_in if step_bytes <= rates.l3_bytes \
                else rates.bw_hi_out
        return rates.bw_lo_in if step_bytes <= rates.l3_bytes \
            else rates.bw_lo_out

    reduce_a = bool(la - lb - out_set)
    reduce_b = bool(lb - la - out_set)
    if out_rank == 0 and k > 1:
        return flops / rates.r_reduce
    if out_rank == 1 and k > 1:
        return flops / rates.r_reduce_out1
    if m == 1 and n == 1:
        bw = rates.bw_mul if step_bytes <= rates.slow_cliff_bytes \
            else rates.bw_mul_out
        c = step_bytes / bw
    else:
        c = flops / rates.r_gemm
        if reduce_a:
            c += 2.0 * _bytes(sorted(la), dom, ext) / bw_traffic()
        if reduce_b:
            c += 2.0 * _bytes(sorted(lb), dom, ext) / bw_traffic()
    floor = rates.step_floor_hi if out_rank >= 3 else \
        rates.step_floor_out2 if out_rank == 2 else 0.0
    return max(c, floor)


def dp_tree_wall(factors, out_labels, dom, ext, rates, step_flop=None):
    """Optimal binary parenthesization minimizing v3 wall time
    (clean-assumed). Mirrors proto.dp_tree's structure: returns
    (wall_cost, flop_cost_of_the_same_tree, tree, result_labels) with
    tree = int | (ltree, rtree). `step_flop` prices one step under the
    truthmill FLOP convention (proto.step_cost), for the Axis-1 cap row."""
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

    from functools import lru_cache as _lc

    @_lc(maxsize=None)
    def best(mask):
        if mask & (mask - 1) == 0:
            return 0.0, 0, None
        bwall, bflop, bsplit = None, None, None
        sub = (mask - 1) & mask
        while sub:
            other = mask ^ sub
            if other and sub < other:
                w1, f1, _ = best(sub)
                w2, f2, _ = best(other)
                la = result_labels(sub)
                lb = result_labels(other)
                step = step_wall_sets(la, lb, result_labels(mask),
                                      dom, ext, rates)
                wall = w1 + w2 + step
                flop = f1 + f2 + (step_flop([la, lb], result_labels(mask),
                                            dom, ext)
                                  if step_flop else 0)
                if bwall is None or wall < bwall:
                    bwall, bflop, bsplit = wall, flop, (sub, other)
            sub = (sub - 1) & mask
        return bwall, bflop, bsplit

    full = (1 << n) - 1

    def build(mask):
        if mask & (mask - 1) == 0:
            return mask.bit_length() - 1
        _, _, (a, b) = best(mask)
        return (build(a), build(b))

    wall, flop, _ = best(full)
    return wall, flop, build(full), result_labels


def einsum_wall(inputs, output, dom, ext, rates, layout="as_written"):
    """v3 wall-time price of one einsum op, seconds. `inputs`/`output` are
    label lists (as-written order); `dom` label->domain; `ext` domain->size.

    layout: "as_written" prices the physical order given (harness-reality
    for unplanned programs); "clean_assumed" prices the post-`--layout
    model` reality the ILP optimizes for: canonical operand order, so
    GEMM-class steps run clean (order-free, set-based). Measured divergence
    of clean_assumed vs harness layout-model pricing: totals 0.78-0.83x on
    adc2/ccsd_gristmill (under-pricing ~20%, ranking-oriented; NOTES.md).
    """
    size_of = {l: ext[dom[l]] for l in dom}
    # Single-input einsum = pure permute (no tree steps).
    if len(inputs) == 1:
        if list(inputs[0]) == list(output):
            return 0.0
        b = _bytes(inputs[0], dom, ext) + _bytes(output, dom, ext)
        if len(output) >= 5:
            bw = rates.bw_hi_in if b <= rates.l3_bytes else rates.bw_hi_out
        else:
            bw = rates.bw_view
        c = b / bw
        return max(c, rates.step_floor_hi) if len(output) >= 3 else c

    if layout == "clean_assumed":
        if len(inputs) == 2:
            return max(
                step_wall_sets(inputs[0], inputs[1], output, dom, ext, rates),
                rates.dispatch)
        return max(dp_tree_wall(inputs, output, dom, ext, rates)[0],
                   rates.dispatch)

    total = 0.0
    for lhs, rhs, out in _tree_steps(inputs, output, size_of):
        sp = _step_plan(lhs, rhs, out, dom, ext)
        flops = 2.0 * sp.m * sp.n * sp.k * sp.batch
        clean = (list(lhs) == sp.target_a) and (list(rhs) == sp.target_b)
        out_rank = len(out)
        max_rank = max(out_rank, len(lhs), len(rhs))
        step_bytes = _bytes(lhs, dom, ext) + _bytes(rhs, dom, ext) \
            + _bytes(out, dom, ext)

        def bw_traffic():
            if max_rank >= 5:
                return rates.bw_hi_in if step_bytes <= rates.l3_bytes \
                    else rates.bw_hi_out
            return rates.bw_lo_in if step_bytes <= rates.l3_bytes \
                else rates.bw_lo_out

        def bw_perm(traffic):
            return rates.bw_lo_in if traffic <= rates.l3_bytes \
                else rates.bw_lo_out

        k_dominant = sp.k >= sp.m * sp.n
        branch = None
        if out_rank == 0 and sp.k > 1:
            # reduction class (thread-flat aligned rate); strided feeds a
            # real permute first, priced by traversal scatter
            if clean:
                c = flops / rates.r_reduce
            else:
                big = max(_bytes(lhs, dom, ext), _bytes(rhs, dom, ext))
                cur, target = (lhs, sp.target_a) if list(lhs) != sp.target_a \
                    else (rhs, sp.target_b)
                desc = 0
                for w0, w1 in zip(cur, cur[1:]):
                    if w0 in target and w1 in target and \
                            target.index(w0) > target.index(w1):
                        desc += 1
                scattered = len(cur) > 1 and desc * 2 >= len(cur) - 1
                bw = rates.bw_hi_out if scattered else bw_perm(2.0 * big)
                c = 2.0 * big / bw + flops / rates.r_reduce
            branch = "reduce"
        elif out_rank == 1 and sp.k > 1:
            c = flops / rates.r_reduce_out1
            branch = "reduce1"
        elif sp.m == 1 and sp.n == 1:
            bw = rates.bw_mul if step_bytes <= rates.slow_cliff_bytes \
                else rates.bw_mul_out
            c = step_bytes / bw
            branch = "mul"
        elif not clean and k_dominant:
            c = step_bytes / bw_traffic() + flops / rates.r_slow
            branch = "kdom"
        elif clean:
            c = flops / rates.r_gemm
            branch = "clean"
        else:
            r = rates.r_slow_in if step_bytes <= rates.slow_cliff_bytes \
                else rates.r_slow
            c = flops / r
            branch = "slow"
        # reduce pre-passes ride the GEMM branches only (the measured
        # reduction rates already include them)
        if branch in ("clean", "slow"):
            if sp.reduce_a:
                c += 2.0 * _bytes(lhs, dom, ext) / bw_traffic()
            if sp.reduce_b:
                c += 2.0 * _bytes(rhs, dom, ext) / bw_traffic()
        if list(out) != sp.canonical_modes and branch in ("clean", "slow"):
            c += _bytes(out, dom, ext) / bw_traffic()
        if branch not in ("reduce", "reduce1"):
            floor = rates.step_floor_hi if out_rank >= 3 else \
                rates.step_floor_out2 if out_rank == 2 else 0.0
            c = max(c, floor)
        total += c
    return max(total, rates.dispatch)


def lincomb_wall(inputs, output, dom, ext, rates):
    """v3 LinComb branch: per-input permute+accumulate traffic + share of
    the output write; per-input floor at rank>=5."""
    out_rank = len(output)
    out_bytes = _bytes(output, dom, ext)
    cost = 0.0
    for labels in inputs:
        b = _bytes(labels, dom, ext)
        traffic = 3.0 * b + out_bytes
        if len(labels) >= 5 or out_rank >= 5:
            bw = rates.bw_hi_in if traffic <= rates.l3_bytes \
                else rates.bw_hi_out
        else:
            bw = rates.bw_lo_in if traffic <= rates.l3_bytes \
                else rates.bw_lo_out
        c = traffic / bw
        if len(labels) >= 5 or out_rank >= 5:
            c = max(c, rates.lincomb_floor_hi)
        cost += c
    return max(cost, rates.dispatch)


def wall_add_unit(elems, rank, rates):
    """Wall cost of one final-lincomb add over `elems` output elements at
    output rank `rank` (the ILP's add-unit accounting, B-007)."""
    bytes_ = elems * 8.0
    traffic = 3.0 * bytes_ + bytes_
    if rank >= 5:
        bw = rates.bw_hi_in if traffic <= rates.l3_bytes else rates.bw_hi_out
    else:
        bw = rates.bw_lo_in if traffic <= rates.l3_bytes else rates.bw_lo_out
    c = traffic / bw
    if rank >= 5:
        c = max(c, rates.lincomb_floor_hi)
    return c


def permute_wall(elems, rank, rates):
    """Wall cost of a standalone output-permute op over `elems` elements at
    rank `rank` (single-input einsum; FLOP-free in the truthmill convention
    but NOT wall-free — the ccd self-check delta was exactly these)."""
    b = 2.0 * elems * 8.0
    if rank >= 5:
        bw = rates.bw_hi_in if b <= rates.l3_bytes else rates.bw_hi_out
    else:
        bw = rates.bw_view
    c = b / bw
    if rank >= 3:
        c = max(c, rates.step_floor_hi)
    return c


# ------------------------------------------------------------ Program JSON

def price_candidate_wall(cand, ext, rates, layout="clean_assumed"):
    """v3 wall-time price of an emitted candidate Program, seconds.
    `layout` defaults to the ILP's convention (clean_assumed); pass
    "as_written" to measure harness-reality divergence."""
    tot = 0.0
    for v in cand["values"]:
        if "Einsum" in v:
            op = v["Einsum"]
            code = op["code"]
            dom = {int(l): d for l, d in code["domains"].items()}
            inputs = code["inputs"]
            tot += einsum_wall(inputs, code["output"], dom, ext, rates,
                               layout=layout)
        else:
            op = v["LinComb"]
            code = op["code"]
            dom = {int(l): d for l, d in code["domains"].items()}
            tot += lincomb_wall(code["inputs"], code["output"], dom, ext,
                                rates)
    return tot
