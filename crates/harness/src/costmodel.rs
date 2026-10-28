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
    /// Effective permute/copy bandwidth, bytes/s (charged read+write).
    pub bw: f64,
    /// Per-op dispatch floor, seconds.
    pub dispatch: f64,
}

/// Calibrated on this box via microbench (v=64, o=16 contraction):
/// t1: clean GEMM 60.4 GF/s, slow path 22.5 GF/s, permute 2.8 GB/s.
/// t16: 220.7 / 158.1 GF/s, 4.1 GB/s.
pub fn rates_for_threads(threads: usize) -> CostRates {
    if threads >= 8 {
        CostRates {
            r_gemm: 220.7e9,
            r_slow: 158.1e9,
            bw: 4.1e9,
            dispatch: 40e-6,
        }
    } else {
        CostRates {
            r_gemm: 60.4e9,
            r_slow: 22.5e9,
            bw: 2.8e9,
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
            let subs_refs: Vec<&[u32]> = phys.iter().map(Vec::as_slice).collect();
            let subs = Subscripts::new(&subs_refs, &output);
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
                let flops =
                    2.0 * gemm.m() as f64 * gemm.n() as f64 * gemm.k() as f64
                        * gemm.batch_shape().iter().product::<usize>() as f64;
                let bytes_of = |labels: &[u32]| {
                    labels.iter().map(|l| size_of[l]).product::<usize>() as f64 * 8.0
                };
                // The backend lowers to canonical GEMM targets; if the
                // physical layout of either operand misses, the whole step
                // drops to the slow path (measured: 22.5 vs 60.4 GF/s t1).
                let clean = lhs == gemm.lhs_target_modes() && rhs == gemm.rhs_target_modes();
                let mut step_cost = if clean {
                    flops / rates.r_gemm
                } else {
                    flops / rates.r_slow
                };
                if out != gemm.canonical_output_modes() {
                    step_cost += 2.0 * bytes_of(out) / rates.bw;
                }
                if plan.lhs_reduce().is_some() {
                    step_cost += 2.0 * bytes_of(lhs) / rates.bw;
                }
                if plan.rhs_reduce().is_some() {
                    step_cost += 2.0 * bytes_of(rhs) / rates.bw;
                }
                cost += step_cost;
            }
            cost.max(rates.dispatch)
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
