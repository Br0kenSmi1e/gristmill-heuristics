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

use crate::layout::Layouts;

/// Calibrated rates for one thread count.
#[derive(Clone, Copy, Debug)]
pub struct CostRates {
    /// Clean-GEMM rate, flops/s.
    pub r_gemm: f64,
    /// Backend slow-path rate when operand layouts miss canonical targets.
    pub r_slow: f64,
    /// Reduction-step rate (step output rank 0, contracted axes present),
    /// operands at canonical targets. THREAD-FLAT by measurement.
    pub r_reduce: f64,
    /// As r_reduce, operands off canonical targets (strided).
    pub r_reduce_strided: f64,
    /// Reduction steps with rank-1 output. At t16 this backend regime is
    /// pathological AND unstable (6-15x slower than t1, 2.6x session
    /// variance): priced pessimistically.
    pub r_reduce_out1: f64,
    /// Effective permute/copy bandwidth, bytes/s (charged read+write).
    pub bw: f64,
    /// Bandwidth for rank>=5 traffic (permutes, batched multiplies):
    /// much slower than mid-rank at t1, faster at t16.
    pub bw_hi: f64,
    /// Fixed floor per rank>=5 step, seconds (measured overhead the
    /// dispatch floor misses: ~1.3 ms on tiny rank-6 ops).
    pub step_floor_hi: f64,
    /// Per-op dispatch floor, seconds.
    pub dispatch: f64,
}

/// Model v2, calibrated on this box against profiles/calibrate-v1.json
/// (git f25f374, trials/m11-calibration.jsonl; two guard-compliant
/// sessions, >=100 ms cases reproduce within 3%).
///   t1:  clean GEMM 57.7 GF/s, slow path 23.0 GF/s, reduce 0.09/0.04 GF/s,
///        rank-1-out reduce 0.05 GF/s, bw 2.8 GB/s, rank>=5 bw 0.75 GB/s.
///   t16: 255.3 / 153.0 GF/s, reduce rates flat (measured: no thread
///        scaling), rank-1-out reduce 0.006 GF/s (unstable, pessimistic),
///        bw 4.1, rank>=5 bw 6.5 GB/s.
pub fn rates_for_threads(threads: usize) -> CostRates {
    if threads >= 8 {
        CostRates {
            r_gemm: 255.3e9,
            r_slow: 153.0e9,
            r_reduce: 0.09e9,
            r_reduce_strided: 0.07e9,
            r_reduce_out1: 0.006e9,
            bw: 4.1e9,
            bw_hi: 6.5e9,
            step_floor_hi: 1.3e-3,
            dispatch: 40e-6,
        }
    } else {
        CostRates {
            r_gemm: 57.7e9,
            r_slow: 23.0e9,
            r_reduce: 0.09e9,
            r_reduce_strided: 0.04e9,
            r_reduce_out1: 0.05e9,
            bw: 2.8e9,
            bw_hi: 0.75e9,
            step_floor_hi: 1.3e-3,
            dispatch: 30e-6,
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
            let out_bytes: f64 = op
                .code
                .output
                .iter()
                .map(|l| size_of[&l.0])
                .product::<usize>() as f64
                * 8.0;
            let mut traffic = out_bytes; // one write
            for labels in &op.code.inputs {
                let bytes: f64 =
                    labels.iter().map(|l| size_of[&l.0]).product::<usize>() as f64 * 8.0;
                // permute/copy + accumulate read
                traffic += 3.0 * bytes;
            }
            (traffic / rates.bw).max(rates.dispatch)
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
    let tree = match ContractionTree::optimize(&subs, &shape_refs) {
        Ok(t) => t,
        Err(_) => return f64::INFINITY,
    };
    let mut cost = 0.0;
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
        let bw_step = if out_rank >= 5 || lhs.len() >= 5 || rhs.len() >= 5 {
            rates.bw_hi
        } else {
            rates.bw
        };
        let dbg = std::env::var("H_MODEL_DEBUG").is_ok();
        let mut step_cost = if out_rank == 0 && gemm.k() > 1 {
            // Reduction-class step: never reaches GEMM (calibrate-v1).
            if clean {
                flops / rates.r_reduce
            } else {
                flops / rates.r_reduce_strided
            }
        } else if out_rank == 1 && gemm.k() > 1 {
            // Rank-1-output reduction; pathological at t16 (see rates).
            flops / rates.r_reduce_out1
        } else if gemm.m() == 1 && gemm.n() == 1 {
            // Pure batched multiply (no contraction): bandwidth-bound.
            (bytes_of(lhs) + bytes_of(rhs) + bytes_of(out)) / bw_step
        } else if clean {
            flops / rates.r_gemm
        } else {
            flops / rates.r_slow
        };
        if out != gemm.canonical_output_modes() && !(gemm.m() == 1 && gemm.n() == 1) {
            // (The batched-multiply branch already charges full traffic.)
            step_cost += 2.0 * bytes_of(out) / bw_step;
        }
        if plan.lhs_reduce().is_some() {
            step_cost += 2.0 * bytes_of(lhs) / bw_step;
        }
        if plan.rhs_reduce().is_some() {
            step_cost += 2.0 * bytes_of(rhs) / bw_step;
        }
        // High-rank steps carry a measured fixed overhead (~1.3 ms at rank
        // >= 5, calibrate-v1 small scales) that the dispatch floor misses.
        if out_rank >= 5 || lhs.len() >= 5 || rhs.len() >= 5 {
            step_cost = step_cost.max(rates.step_floor_hi);
        }
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
