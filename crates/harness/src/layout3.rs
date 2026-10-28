//! Model-driven layout assignment (v3): replace the constant-penalty heuristic
//! with the analytic backend cost model. Entities (input blocks and SSA
//! intermediates) each choose a physical axis order; the choice minimizes
//! total predicted wall time. Solved by iterated conditional modes over
//! per-entity candidates {natural} ∪ {per-use ideals}, with local cost
//! deltas (an op's cost only depends on the layouts of the entities it
//! touches).

use std::collections::{BTreeMap, BTreeSet};

use truthmill::{Problem, Program, ValueDef, ValueRef};

use crate::costmodel::{op_cost_virtual, CostRates};
use crate::layout::Layouts;
use crate::layout2::{ideal_binary_arg, ideal_lincomb_arg, ideal_producer, union_flops, ValueLayouts};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Entity {
    Block(u32, Vec<u32>),
    Value(usize),
}

pub fn plan_layouts_model(
    problem: &Problem,
    program: &Program,
    rates: &CostRates,
    passes: usize,
) -> (Layouts, ValueLayouts) {
    // Collect per-entity uses: (touching ops, candidate ideal perms, weight).
    let mut uses: BTreeMap<Entity, Vec<(usize, Vec<usize>)>> = BTreeMap::new();
    let mut weight: BTreeMap<Entity, f64> = BTreeMap::new();
    for (id, definition) in program.values.iter().enumerate() {
        let (code, args, is_einsum) = match definition {
            ValueDef::Einsum(op) => (&op.code, &op.args, true),
            ValueDef::LinComb(op) => (&op.code, &op.args, false),
        };
        let w = union_flops(code, &problem.sizes);
        // Producer use of this op's value.
        if is_einsum {
            if let Some(ideal) = ideal_producer(code) {
                uses.entry(Entity::Value(id)).or_default().push((id, ideal));
            }
        }
        *weight.entry(Entity::Value(id)).or_default() += w;
        for (pos, (arg, labels)) in args.iter().zip(&code.inputs).enumerate() {
            let ideal = if is_einsum {
                ideal_binary_arg(code, pos)
            } else {
                ideal_lincomb_arg(code, pos)
            };
            let Some(ideal) = ideal else { continue };
            let entity = match arg {
                ValueRef::Tensor(t) => {
                    let pattern: Vec<u32> = labels.iter().map(|l| code.domains[l].0).collect();
                    Entity::Block(t.0, pattern)
                }
                ValueRef::Value(v) => Entity::Value(v.0 as usize),
            };
            uses.entry(entity.clone()).or_default().push((id, ideal));
            *weight.entry(entity).or_default() += w;
        }
    }

    // Current layout state (identity = natural when absent).
    let mut block_layouts = Layouts::new();
    let mut value_layouts = ValueLayouts::new();
    let rank_of = |e: &Entity| -> usize {
        match e {
            Entity::Block(_, pat) => pat.len(),
            Entity::Value(id) => {
                let code = match &program.values[*id] {
                    ValueDef::Einsum(op) => &op.code,
                    ValueDef::LinComb(op) => &op.code,
                };
                code.output.len()
            }
        }
    };
    let get_perm = |e: &Entity,
                    bl: &Layouts,
                    vl: &ValueLayouts|
     -> Vec<usize> {
        match e {
            Entity::Block(t, pat) => bl
                .get(&(*t, pat.clone()))
                .cloned()
                .unwrap_or_else(|| (0..pat.len()).collect()),
            Entity::Value(id) => vl
                .get(id)
                .cloned()
                .unwrap_or_else(|| (0..rank_of(e)).collect()),
        }
    };

    // Ops touched by each entity (for local cost deltas).
    let mut touch: BTreeMap<Entity, BTreeSet<usize>> = BTreeMap::new();
    for (e, us) in &uses {
        touch.insert(e.clone(), us.iter().map(|(op, _)| *op).collect());
    }

    // Entities in decreasing total weight.
    let mut entities: Vec<Entity> = uses.keys().cloned().collect();
    entities.sort_by(|a, b| weight[b].partial_cmp(&weight[a]).unwrap());

    let local_cost = |ops: &BTreeSet<usize>,
                      bl: &Layouts,
                      vl: &ValueLayouts|
     -> f64 {
        ops.iter()
            .map(|&id| {
                op_cost_virtual(&program.values[id], id, &problem.sizes, bl, vl, rates)
            })
            .sum()
    };

    for _ in 0..passes {
        let mut changed = false;
        for e in &entities {
            let rank = rank_of(e);
            let mut candidates: Vec<Vec<usize>> = vec![(0..rank).collect()];
            for (_, ideal) in &uses[e] {
                if !candidates.contains(ideal) {
                    candidates.push(ideal.clone());
                }
            }
            let ops = &touch[e];
            let mut best_perm = get_perm(e, &block_layouts, &value_layouts);
            let mut best_cost = local_cost(ops, &block_layouts, &value_layouts);
            for cand in &candidates[1..] {
                // Tentatively set.
                match e {
                    Entity::Block(t, pat) => {
                        block_layouts.insert((t.clone(), pat.clone()), cand.clone());
                    }
                    Entity::Value(id) => {
                        value_layouts.insert(*id, cand.clone());
                    }
                }
                let cost = local_cost(ops, &block_layouts, &value_layouts);
                if cost < best_cost * (1.0 - 1e-9) {
                    best_cost = cost;
                    best_perm = cand.clone();
                    changed = true;
                }
            }
            // Commit best.
            match e {
                Entity::Block(t, pat) => {
                    if best_perm.iter().enumerate().all(|(k, &p)| k == p) {
                        block_layouts.remove(&(*t, pat.clone()));
                    } else {
                        block_layouts.insert((*t, pat.clone()), best_perm);
                    }
                }
                Entity::Value(id) => {
                    if best_perm.iter().enumerate().all(|(k, &p)| k == p) {
                        value_layouts.remove(id);
                    } else {
                        value_layouts.insert(*id, best_perm);
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    (block_layouts, value_layouts)
}
