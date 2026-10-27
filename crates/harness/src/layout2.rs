//! Layout planner v2: uniform cost-model over storage entities.
//!
//! Every tensor in the schedule — input block or SSA intermediate — is an
//! entity with a natural axis order and a set of *uses*: consumer uses (op,
//! arg position) plus, for intermediates, the producer use (the output slot).
//! Each use has an ideal storage order; each entity picks, among {natural} +
//! all use-ideals, the layout minimizing sum of use weights times a mismatch
//! penalty. Blocks are physically permuted at setup; intermediates are
//! reordered at the producer with consumers' labels rewritten positionally.
//!
//! The mismatch penalty is a learnable constant (HL target), initially from
//! the microbenchmark (clean GEMM vs layout-hostile: ~2.7x at t1).

use std::collections::BTreeMap;

use truthmill::{Code, DomainId, IndexId, Problem, Program, ValueDef, ValueRef};

use crate::layout::Layouts;

/// Chosen permutation per SSA value (output reorder), same convention as
/// block layouts: physical axis k = natural axis perm[k].
pub type ValueLayouts = BTreeMap<usize, Vec<usize>>;

/// One use of a storage entity.
struct Use {
    weight: f64,
    ideal: Vec<usize>,
}

fn union_flops(code: &Code, sizes: &BTreeMap<DomainId, u64>) -> f64 {
    let mut labels: Vec<IndexId> = code.output.clone();
    for input in &code.inputs {
        for l in input {
            if !labels.contains(l) {
                labels.push(*l);
            }
        }
    }
    labels
        .iter()
        .map(|l| sizes[&code.domains[l]] as f64)
        .product()
}

/// Ideal storage perm for argument `arg_pos` of a binary einsum
/// (free-first for A, contracted-first for B), in the entity's natural axis
/// positions.
fn ideal_binary_arg(code: &Code, arg_pos: usize) -> Option<Vec<usize>> {
    if code.inputs.len() != 2 {
        return None;
    }
    let a = &code.inputs[0];
    let b = &code.inputs[1];
    let contracted: Vec<IndexId> = a.iter().filter(|l| b.contains(l)).cloned().collect();
    let mine = &code.inputs[arg_pos];
    let free: Vec<usize> = mine
        .iter()
        .enumerate()
        .filter(|(_, l)| !contracted.contains(l))
        .map(|(k, _)| k)
        .collect();
    let contr: Vec<usize> = mine
        .iter()
        .enumerate()
        .filter(|(_, l)| contracted.contains(l))
        .map(|(k, _)| k)
        .collect();
    Some(if arg_pos == 0 {
        free.into_iter().chain(contr).collect()
    } else {
        contr.into_iter().chain(free).collect()
    })
}

/// Ideal producer-output perm for a binary einsum: [A-free..., B-free...]
/// as positions into the natural output order.
fn ideal_producer(code: &Code) -> Option<Vec<usize>> {
    if code.inputs.len() != 2 {
        return None;
    }
    let a = &code.inputs[0];
    let b = &code.inputs[1];
    let contracted: Vec<IndexId> = a.iter().filter(|l| b.contains(l)).cloned().collect();
    let want: Vec<IndexId> = a
        .iter()
        .filter(|l| !contracted.contains(l))
        .chain(b.iter().filter(|l| !contracted.contains(l)))
        .cloned()
        .collect();
    if want.len() != code.output.len() {
        return None;
    }
    want.iter()
        .map(|l| code.output.iter().position(|o| o == l))
        .collect()
}

/// Ideal arg perm for a LinComb use: make the arg's storage order match the
/// LinComb's output order.
fn ideal_lincomb_arg(code: &Code, arg_pos: usize) -> Option<Vec<usize>> {
    let labels = &code.inputs[arg_pos];
    // storage axis k should hold the axis whose label is output[k];
    // perm[k] = position of output[k]'s label in this arg's labels
    let mut perm = vec![0usize; labels.len()];
    for (k, out_label) in code.output.iter().enumerate() {
        perm[k] = labels.iter().position(|l| l == out_label)?;
    }
    Some(perm)
}

fn pick(uses: &[Use], rank: usize, penalty: f64) -> Vec<usize> {
    let natural: Vec<usize> = (0..rank).collect();
    let mut candidates: Vec<Vec<usize>> = vec![natural.clone()];
    for u in uses {
        if !candidates.contains(&u.ideal) {
            candidates.push(u.ideal.clone());
        }
    }
    let mut best = natural;
    let mut best_cost = f64::INFINITY;
    for cand in &candidates {
        let cost: f64 = uses
            .iter()
            .map(|u| {
                let p = if *cand == u.ideal { 1.0 } else { penalty };
                u.weight * p
            })
            .sum();
        if cost < best_cost {
            best_cost = cost;
            best = cand.clone();
        }
    }
    best
}

/// Plan layouts for input blocks and intermediate values together.
pub fn plan_layouts_v2(
    problem: &Problem,
    program: &Program,
    penalty: f64,
) -> (Layouts, ValueLayouts) {
    let mut block_uses: BTreeMap<(u32, Vec<u32>), Vec<Use>> = BTreeMap::new();
    let mut value_uses: BTreeMap<usize, Vec<Use>> = BTreeMap::new();
    let mut value_rank: BTreeMap<usize, usize> = BTreeMap::new();

    for (id, definition) in program.values.iter().enumerate() {
        let (code, args) = match definition {
            ValueDef::Einsum(op) => (&op.code, &op.args),
            ValueDef::LinComb(op) => (&op.code, &op.args),
        };
        let weight = union_flops(code, &problem.sizes);
        let is_einsum = matches!(definition, ValueDef::Einsum(_));
        // Producer use.
        if is_einsum {
            value_rank.insert(id, code.output.len());
            if let Some(ideal) = ideal_producer(code) {
                value_uses.entry(id).or_default().push(Use { weight, ideal });
            }
        } else {
            value_rank.insert(id, code.output.len());
        }
        // Consumer uses.
        for (pos, (arg, labels)) in args.iter().zip(&code.inputs).enumerate() {
            let ideal = if is_einsum {
                ideal_binary_arg(code, pos)
            } else {
                ideal_lincomb_arg(code, pos)
            };
            let Some(ideal) = ideal else { continue };
            match arg {
                ValueRef::Tensor(tensor) => {
                    let pattern: Vec<u32> = labels.iter().map(|l| code.domains[l].0).collect();
                    block_uses
                        .entry((tensor.0, pattern))
                        .or_default()
                        .push(Use { weight, ideal });
                }
                ValueRef::Value(v) => {
                    value_uses
                        .entry(v.0 as usize)
                        .or_default()
                        .push(Use { weight, ideal });
                }
            }
        }
    }

    let blocks = block_uses
        .into_iter()
        .map(|(key, uses)| {
            let rank = uses[0].ideal.len();
            (key, pick(&uses, rank, penalty))
        })
        .collect();
    let output_ids: std::collections::BTreeSet<usize> =
        program.outputs.iter().map(|o| o.0 as usize).collect();
    let values = value_uses
        .into_iter()
        .filter_map(|(id, uses)| {
            let rank = value_rank[&id];
            // Program outputs are pinned to their natural order: the oracle
            // (and downstream consumers) compare against reference layouts.
            if output_ids.contains(&id) {
                return None;
            }
            let chosen = pick(&uses, rank, penalty);
            if chosen.iter().enumerate().any(|(k, &p)| k != p) {
                Some((id, chosen))
            } else {
                None
            }
        })
        .collect();
    (blocks, values)
}
