//! White-box analytic cost model for tenferro einsum execution.
//!
//! Form (TCE/COGENT tradition: analytic, calibrated, used for RANKING):
//!   cost(op) = sum over the backend's own contraction-tree steps of
//!       GEMM(2*m*n*k*batch / R)                       if operand layouts match
//!                                                    the plan's canonical targets
//!       min(slow-path(flops / R_slow),               if mismatched: whichever
//!           permute(2*bytes/BW) + GEMM)              the backend would face
//!       + final output permute (2*bytes/BW) when the requested output order
//!         is not the plan's canonical [lo, ro, batch] order
//!       + reduce/diag pre-passes (2*bytes/BW each)
//!   floored at `dispatch` per op (small-op floor).
//!
//! The lowering rules (canonical targets, tree structure) are read from
//! tenferro's public planning API (`ContractionTree::optimize`), so the model
//! is faithful to the backend by construction; only the four rates are
//! calibrated (microbench, committed values below).

use std::collections::BTreeMap;

use tenferro_einsum::{ContractionTree, Subscripts};
use truthmill::{DomainId, IndexId, Program, ValueDef, ValueRef};

/// Deterministic contraction pairing: a faithful port of tenferro-einsum
/// 0.7.1's private `optimize_self_greedy_pairs` (planning/tree.rs) —
/// repeatedly contract the pair whose intermediate is smallest, ties to the
/// lexicographically first pair.
///
/// Why not ContractionTree::optimize: for >=3 operands it routes through
/// omeco TreeSA, whose greedy initialization iterates HashMap-backed
/// incidence lists, so contraction-cost ties resolve in per-process random
/// order (F-028: same computation scored 0.397 s vs 0.621 s across
/// processes on mp3). The model must be a pure function of the program.
fn det_greedy_pairs(inputs: &[Vec<u32>], output: &[u32], size_of: &BTreeMap<u32, usize>) -> Vec<(usize, usize)> {
    let input_count = inputs.len();
    if input_count <= 1 {
        return Vec::new();
    }
    let mut available: Vec<usize> = (0..input_count).collect();
    let mut operand_subs: Vec<Vec<u32>> = inputs.to_vec();
    let mut pairs = Vec::new();
    // Count, per label, how many live operands (set membership, matching
    // upstream) plus the output still need it.
    let mut needed: BTreeMap<u32, usize> = BTreeMap::new();
    for &l in output {
        *needed.entry(l).or_insert(0) += 1;
    }
    for subs in &operand_subs {
        let mut seen = Vec::new();
        for &l in subs {
            if !seen.contains(&l) {
                seen.push(l);
                *needed.entry(l).or_insert(0) += 1;
            }
        }
    }
    // Intermediate subscripts: labels of left/right still needed elsewhere,
    // in left-then-right first-appearance order (matches upstream).
    let contains = |subs: &[u32], l: u32| subs.iter().any(|&x| x == l);
    let intermediate = |left: &[u32], right: &[u32], needed: &BTreeMap<u32, usize>| -> Vec<u32> {
        let mut out = Vec::new();
        for &l in left.iter().chain(right.iter()) {
            let selected = contains(left, l) as usize + contains(right, l) as usize;
            let still_needed = needed.get(&l).copied().unwrap_or(0) > selected;
            if still_needed && !out.contains(&l) {
                out.push(l);
            }
        }
        out
    };
    let remove_counts = |needed: &mut BTreeMap<u32, usize>, subs: &[u32]| {
        let mut seen = Vec::new();
        for &l in subs {
            if seen.contains(&l) {
                continue;
            }
            seen.push(l);
            if let Some(c) = needed.get_mut(&l) {
                *c -= 1;
                if *c == 0 {
                    needed.remove(&l);
                }
            }
        }
    };
    while available.len() > 1 {
        let (mut best_i, mut best_j, mut best_cost) = (0, 1, usize::MAX);
        for i in 0..available.len() {
            for j in (i + 1)..available.len() {
                let (li, lj) = (available[i], available[j]);
                let subs = intermediate(&operand_subs[li], &operand_subs[lj], &needed);
                let cost = subs
                    .iter()
                    .map(|l| size_of.get(l).copied().unwrap_or(1))
                    .product::<usize>()
                    .max(1);
                if cost < best_cost {
                    best_cost = cost;
                    best_i = i;
                    best_j = j;
                }
            }
        }
        let (left, right) = (available[best_i], available[best_j]);
        pairs.push((left, right));
        // Intermediate labels are computed with CURRENT counts (upstream
        // order), then the pair's counts are removed and the
        // intermediate's added.
        let new_subs = intermediate(&operand_subs[left], &operand_subs[right], &needed);
        remove_counts(&mut needed, &operand_subs[left]);
        remove_counts(&mut needed, &operand_subs[right]);
        let mut seen = Vec::new();
        for &l in &new_subs {
            if !seen.contains(&l) {
                seen.push(l);
                *needed.entry(l).or_insert(0) += 1;
            }
        }
        let new_idx = operand_subs.len();
        operand_subs.push(new_subs);
        available.remove(best_j);
        available.remove(best_i);
        available.push(new_idx);
    }
    pairs
}

use crate::layout::Layouts;

/// Calibrated rates for one thread count (model v3, release regime).
#[derive(Clone, Copy, Debug)]
pub struct CostRates {
    /// Clean-GEMM rate, flops/s.
    pub r_gemm: f64,
    /// Slow-path GEMM rate, working set beyond the L3-effective cliff.
    pub r_slow: f64,
    /// Slow-path GEMM rate, cache-resident working set (2.4x faster).
    pub r_slow_in: f64,
    /// Reduction-step rate (step output rank 0, contracted axes present),
    /// operands at canonical targets. THREAD-FLAT by measurement.
    pub r_reduce: f64,
    /// Reduction steps with rank-1 output. At t16 this regime is
    /// pathological AND unstable (E-core scheduling on the hybrid CPU):
    /// priced pessimistically.
    pub r_reduce_out1: f64,
    /// Bandwidth for rank<=4 traffic whose working set fits L3, bytes/s.
    pub bw_lo_in: f64,
    /// Bandwidth for rank<=4 traffic exceeding L3, bytes/s.
    pub bw_lo_out: f64,
    /// Bandwidth for rank>=5 traffic, working set within L3-effective.
    pub bw_hi_in: f64,
    /// Bandwidth for rank>=5 traffic beyond L3-effective (scattered copy).
    pub bw_hi_out: f64,
    /// Bandwidth for batched elementwise multiply steps, in-cliff working
    /// set (streaming, fast) / beyond the reuse cliff.
    pub bw_mul: f64,
    pub bw_mul_out: f64,
    /// Effective rate for rank<=4 single-input permutes (largely view-based).
    pub bw_view: f64,
    /// L3-effective working-set threshold in bytes (measured knee: in-cache
    /// rates hold to ~2x the physical 20 MB L3 on i5-13400F thanks to
    /// streaming reuse; the cliff is between 21 and 43 MB of traffic).
    pub l3_bytes: f64,
    /// Cache cliff for the slow-GEMM compute rate: deep k-reuse keeps the
    /// rate high well past L3 (measured: in-rate at 48 MB step traffic,
    /// out-rate at 145 MB), bytes.
    pub slow_cliff_bytes: f64,
    /// Per-step floor for materializing steps with output rank >= 3, seconds.
    pub step_floor_hi: f64,
    /// Per-step floor for materializing steps with output rank 2, seconds.
    pub step_floor_out2: f64,
    /// Per-input floor for rank>=5 LinComb permutes (permute_r6), seconds.
    pub lincomb_floor_hi: f64,
    /// Per-op dispatch floor, seconds.
    pub dispatch: f64,
}

/// Model v3, calibrated on this box (i5-13400F, 6P+4E hybrid, L3 20 MB)
/// against profiles/calibrate-v2.json — release-profile build, merged from
/// two clean full-grid runs (git 737c26d, loads 0.46/0.63). Cross-session
/// t1 anchor variance on the hybrid CPU is 20-30% (P/E lottery); the 2x CI
/// gate absorbs it. Release vs debug changed the reduction class ~50x, so
/// v1/v2 rates are NOT comparable to v3 (F-033/F-036).
pub fn rates_for_threads(threads: usize) -> CostRates {
    if threads >= 8 {
        CostRates {
            r_gemm: 366.5e9,
            r_slow: 143.7e9,
            r_slow_in: 256.7e9,
            r_reduce: 4.15e9,
            r_reduce_out1: 0.004e9,
            bw_lo_in: 21.5e9,
            bw_lo_out: 6.9e9,
            bw_hi_in: 7.2e9,
            bw_hi_out: 4.1e9,
            bw_mul: 25.1e9,
            bw_mul_out: 15.4e9,
            bw_view: 19.0e9,
            l3_bytes: 40e6,
            slow_cliff_bytes: 100e6,
            step_floor_hi: 1e-4,
            step_floor_out2: 1e-4,
            lincomb_floor_hi: 8e-5,
            dispatch: 15e-6,
        }
    } else {
        CostRates {
            r_gemm: 64.0e9,
            r_slow: 16.9e9,
            r_slow_in: 39.7e9,
            r_reduce: 4.2e9,
            r_reduce_out1: 0.24e9,
            bw_lo_in: 8.0e9,
            bw_lo_out: 2.6e9,
            bw_hi_in: 7.4e9,
            bw_hi_out: 2.8e9,
            bw_mul: 17.5e9,
            bw_mul_out: 6.9e9,
            bw_view: 10.2e9,
            l3_bytes: 40e6,
            slow_cliff_bytes: 100e6,
            step_floor_hi: 1e-4,
            step_floor_out2: 1e-4,
            lincomb_floor_hi: 8e-5,
            dispatch: 15e-6,
        }
    }
}

fn label_sizes(
    code_domains: &BTreeMap<IndexId, DomainId>,
    sizes: &BTreeMap<DomainId, u64>,
) -> BTreeMap<u32, usize> {
    code_domains
        .iter()
        .map(|(l, d)| (l.0, sizes[d] as usize))
        .collect()
}

/// Predicted cost of one op in seconds.
pub fn op_cost(
    definition: &ValueDef,
    sizes: &BTreeMap<DomainId, u64>,
    block_layouts: &Layouts,
    rates: &CostRates,
) -> f64 {
    op_cost_virtual(definition, usize::MAX, sizes, block_layouts, &ValueLayouts::new(), rates)
}

use crate::layout2::ValueLayouts;

/// As `op_cost`, but with explicit intermediate-value layouts (used by the
/// layout solver, which evaluates candidates without rewriting the program).
pub fn op_cost_virtual(
    definition: &ValueDef,
    self_id: usize,
    sizes: &BTreeMap<DomainId, u64>,
    block_layouts: &Layouts,
    value_layouts: &ValueLayouts,
    rates: &CostRates,
) -> f64 {
    match definition {
        ValueDef::Einsum(op) => {
            // Physical input label orders (block args permuted by layout).
            let mut phys: Vec<Vec<u32>> = Vec::new();
            for (arg, labels) in op.args.iter().zip(&op.code.inputs) {
                match arg {
                    ValueRef::Tensor(t) => {
                        let pattern: Vec<u32> =
                            labels.iter().map(|l| op.code.domains[l].0).collect();
                        let perm = block_layouts.get(&(t.0, pattern));
                        match perm {
                            Some(p) => phys
                                .push(p.iter().map(|&k| labels[k].0).collect()),
                            None => phys.push(labels.iter().map(|l| l.0).collect()),
                        }
                    }
                    ValueRef::Value(v) => {
                        // Producer physical order = its perm applied to the
                        // consumer's positional labels.
                        match value_layouts.get(&(v.0 as usize)) {
                            Some(p) => phys
                                .push(p.iter().map(|&k| labels[k].0).collect()),
                            None => phys.push(labels.iter().map(|l| l.0).collect()),
                        }
                    }
                }
            }
            let mut output: Vec<u32> = op.code.output.iter().map(|l| l.0).collect();
            if let Some(p) = value_layouts.get(&self_id) {
                let old = output.clone();
                for (k, &pk) in p.iter().enumerate() {
                    output[k] = old[pk];
                }
            }
            let size_of = label_sizes(&op.code.domains, sizes);
            einsum_cost_model(&phys, &output, &size_of, rates).max(rates.dispatch)
        }
        ValueDef::LinComb(op) => {
            let size_of = label_sizes(&op.code.domains, sizes);
            let out_rank = op.code.output.len();
            let out_bytes: f64 = op
                .code
                .output
                .iter()
                .map(|l| size_of[&l.0])
                .product::<usize>() as f64
                * 8.0;
            let mut cost = 0.0;
            for labels in &op.code.inputs {
                let bytes: f64 =
                    labels.iter().map(|l| size_of[&l.0]).product::<usize>() as f64 * 8.0;
                // permute/copy + accumulate read + share of the output write
                let traffic = 3.0 * bytes + out_bytes;
                let bw = if labels.len() >= 5 || out_rank >= 5 {
                    if traffic <= rates.l3_bytes { rates.bw_hi_in } else { rates.bw_hi_out }
                } else if traffic <= rates.l3_bytes {
                    rates.bw_lo_in
                } else {
                    rates.bw_lo_out
                };
                let mut c = traffic / bw;
                // Measured per-input permute floor at rank>=5 (permute_r6).
                if labels.len() >= 5 || out_rank >= 5 {
                    c = c.max(rates.lincomb_floor_hi);
                }
                cost += c;
            }
            cost.max(rates.dispatch)
        }
    }
}


/// Model v2 core: price one einsum from physical input label orders, the
/// output label order, and per-label extents. Walks the backend's own
/// contraction tree and prices each step by its measured class (F-004,
/// calibrate-v1):
///   - reduction steps (output rank <= 1 with contracted axes): the
///     backend's reduction path, 250-3000x below clean GEMM, thread-flat;
///   - pure batched multiplies (m == n == 1): bandwidth-bound;
///   - otherwise clean/slow GEMM as in v1, plus permute/reduce pre-pass
///     traffic charged at rank-dependent bandwidth.
pub fn einsum_cost_model(
    phys: &[Vec<u32>],
    output: &[u32],
    size_of: &BTreeMap<u32, usize>,
    rates: &CostRates,
) -> f64 {
    let subs_refs: Vec<&[u32]> = phys.iter().map(Vec::as_slice).collect();
    let subs = Subscripts::new(&subs_refs, output);
    let shapes: Vec<Vec<usize>> = phys
        .iter()
        .map(|labels| labels.iter().map(|l| size_of[l]).collect())
        .collect();
    let shape_refs: Vec<&[usize]> = shapes.iter().map(Vec::as_slice).collect();
    // Deterministic planning (F-028): ContractionTree::optimize routes >=3
    // operands through omeco TreeSA, whose tie-breaking is per-process
    // random; pin the tree with the ported self-greedy pairing instead.
    let pairs = det_greedy_pairs(phys, output, size_of);
    let tree = match ContractionTree::from_pairs(&subs, &shape_refs, &pairs) {
        Ok(t) => t,
        Err(_) => return f64::INFINITY,
    };
    let mut cost = 0.0;
    // Single-input einsum = pure permute: no contraction steps exist, so
    // the tree walk prices nothing. Rank<=4 permutes are largely
    // view-based (measured 40-80 GB/s effective); rank>=5 permutes are
    // real scattered copies (permute_r6: ~3 GB/s).
    if phys.len() == 1 {
        if phys[0] == output {
            return 0.0;
        }
        let bytes: f64 = (phys[0].iter().map(|l| size_of[l]).product::<usize>()
            + output.iter().map(|l| size_of[l]).product::<usize>())
            as f64
            * 8.0;
        let bw = if output.len() >= 5 {
            if bytes <= rates.l3_bytes { rates.bw_hi_in } else { rates.bw_hi_out }
        } else {
            rates.bw_view
        };
        let c = bytes / bw;
        return if output.len() >= 3 {
            c.max(rates.step_floor_hi)
        } else {
            c
        };
    }
    for step in 0..tree.step_count() {
        let Some(plan) = tree.step_plan(step) else {
            continue;
        };
        let Some((lhs, rhs, out)) = tree.step_subscripts(step) else {
            continue;
        };
        let gemm = plan.gemm();
        let flops = 2.0
            * gemm.m() as f64
            * gemm.n() as f64
            * gemm.k() as f64
            * gemm.batch_shape().iter().product::<usize>() as f64;
        let bytes_of =
            |labels: &[u32]| labels.iter().map(|l| size_of[l]).product::<usize>() as f64 * 8.0;
        let clean = lhs == gemm.lhs_target_modes() && rhs == gemm.rhs_target_modes();
        let out_rank = out.len();
        let max_rank = out_rank.max(lhs.len()).max(rhs.len());
        let step_bytes = bytes_of(lhs) + bytes_of(rhs) + bytes_of(out);
        // Bandwidth for traffic-bearing charges: rank>=5 traffic is
        // scattered and cache-insensitive (bw_hi); rank<=4 splits at the
        // measured L3 cliff (calibrate-v2: kheavy 8.7->2.7 GB/s past 20 MB).
        let bw_traffic = if max_rank >= 5 {
            if step_bytes <= rates.l3_bytes { rates.bw_hi_in } else { rates.bw_hi_out }
        } else if step_bytes <= rates.l3_bytes {
            rates.bw_lo_in
        } else {
            rates.bw_lo_out
        };
        // Permutes feeding reductions are fused strided passes (stream
        // once, no repack), priced at the low-rank bandwidth even when the
        // operands are high-rank (calibrate-v2 chain_step2 vs permute_r6).
        // The pass footprint is the charged traffic itself (2x the permuted
        // operand), not the whole step's tensor set.
        let bw_perm = |traffic: f64| {
            if traffic <= rates.l3_bytes {
                rates.bw_lo_in
            } else {
                rates.bw_lo_out
            }
        };
        let dbg = std::env::var("H_MODEL_DEBUG").is_ok();
        let k_dominant = (gemm.k() as f64) >= (gemm.m() as f64) * (gemm.n() as f64);
        // Reduce pre-passes (partial sums before GEMM) are real traffic on
        // GEMM branches; the measured reduction rates already include them.
        let mut pre_pass = 0.0;
        if plan.lhs_reduce().is_some() {
            pre_pass += 2.0 * bytes_of(lhs) / bw_traffic;
        }
        if plan.rhs_reduce().is_some() {
            pre_pass += 2.0 * bytes_of(rhs) / bw_traffic;
        }
        let mut step_cost = if out_rank == 0 && gemm.k() > 1 {
            // Reduction-class step. Strided operands force a real permute
            // first (calibrate-v2): price as permute + aligned reduction.
            // The measured reduce rate INCLUDES any reduce pre-passes, so
            // none are charged on this branch.
            if clean {
                flops / rates.r_reduce
            } else {
                // The feeding permute's bandwidth depends on its SCATTER:
                // adjacent-label runs stream at bw_lo (fabdec), reversals
                // jump cache lines at bw_hi_out even when small (fedcba).
                let big = bytes_of(lhs).max(bytes_of(rhs));
                let scattered = {
                    let (cur, target) = if lhs != gemm.lhs_target_modes() {
                        (lhs, gemm.lhs_target_modes())
                    } else {
                        (rhs, gemm.rhs_target_modes())
                    };
                    // Scattered = mostly-descending traversal of the
                    // canonical order (reverse-like jumps thrash cache
                    // lines); a mostly-ascending traversal streams.
                    let mut desc = 0usize;
                    for w in cur.windows(2) {
                        let pa = target.iter().position(|x| *x == w[0]);
                        let pb = target.iter().position(|x| *x == w[1]);
                        if matches!((pa, pb), (Some(a), Some(b)) if a > b) {
                            desc += 1;
                        }
                    }
                    cur.len() > 1 && desc * 2 >= cur.len() - 1
                };
                let bw = if scattered {
                    rates.bw_hi_out
                } else {
                    bw_perm(2.0 * big)
                };
                2.0 * big / bw + flops / rates.r_reduce
            }
        } else if out_rank == 1 && gemm.k() > 1 {
            // Rank-1-output reduction; pathological at t16 (see rates).
            // Pre-passes likewise included in the measured rate.
            flops / rates.r_reduce_out1
        } else if gemm.m() == 1 && gemm.n() == 1 {
            // Pure batched multiply (no contraction): streaming bandwidth,
            // cache-split like the slow-GEMM rate (51 MB in, 573 MB out).
            let bw = if step_bytes <= rates.slow_cliff_bytes {
                rates.bw_mul
            } else {
                rates.bw_mul_out
            };
            step_bytes / bw
        } else if !clean && k_dominant {
            // F-016 class: k-dominant non-clean contraction (kheavy):
            // permute-bound traffic + slow-GEMM compute.
            step_bytes / bw_traffic + flops / rates.r_slow
        } else if clean {
            flops / rates.r_gemm + pre_pass
        } else {
            // Slow GEMM splits at the cache cliff like everything else
            // (calibrate-v2 ccd_slowpath_incache: 39.7 vs 16.9 GF/s t1).
            let r = if step_bytes <= rates.slow_cliff_bytes {
                rates.r_slow_in
            } else {
                rates.r_slow
            };
            flops / r + pre_pass
        };
        if out != gemm.canonical_output_modes()
            && !(gemm.m() == 1 && gemm.n() == 1)
            && (clean || !k_dominant)
        {
            // GEMM branches price compute only; noncanonical output order
            // costs the output write through the permute path (the slow
            // rate already includes internal operand handling).
            step_cost += bytes_of(out) / bw_traffic;
        }
        // Materializing-step floors (measured small-size overheads);
        // reduction steps (out rank <= 1) are exempt — they stay fast
        // at tiny sizes on release.
        let floor = if out_rank >= 3 {
            rates.step_floor_hi
        } else if out_rank == 2 {
            rates.step_floor_out2
        } else {
            0.0
        };
        step_cost = step_cost.max(floor);
        cost += step_cost;
        if dbg {
            eprintln!(
                "  [model] step {step}: {lhs:?},{rhs:?}->{out:?} m={} n={} k={} clean={clean} cost {:.3} ms",
                gemm.m(),
                gemm.n(),
                gemm.k(),
                step_cost * 1e3,
            );
        }
    }
    cost
}

/// Predicted per-op costs for a whole program (seconds), aligned with value ids.
pub fn program_costs(
    program: &Program,
    sizes: &BTreeMap<DomainId, u64>,
    block_layouts: &Layouts,
    rates: &CostRates,
) -> Vec<f64> {
    program
        .values
        .iter()
        .map(|def| op_cost(def, sizes, block_layouts, rates))
        .collect()
}
