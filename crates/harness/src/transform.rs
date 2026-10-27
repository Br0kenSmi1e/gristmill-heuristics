//! Program -> Program transforms (the baby scheduler).
//!
//! v0 heuristic: `greedy_factor` breaks n-ary einsums (k >= 3 args) into
//! binary contractions, greedily pairing the two arguments whose contraction
//! minimizes the pairwise FLOP proxy (product of union index sizes), with
//! tie-break on smaller intermediate. Intermediate outputs drop indices that
//! appear in neither the remaining args nor the final output.
//!
//! Every transform is checked downstream by truthmill::verify (exact) and the
//! numerical oracle, so heuristic bugs are caught, never silent.

use std::collections::BTreeMap;

use truthmill::{Code, Coefficient, DomainId, Einsum, IndexId, Program, ValueDef, ValueRef};

/// One argument's index structure: (domain) per axis, plus identity labels
/// that are shared across the whole original einsum.
#[derive(Clone, Debug)]
struct ArgView {
    reference: ValueRef,
    labels: Vec<u32>,
    domains: Vec<DomainId>,
}

fn union_cost(a: &ArgView, b: &ArgView, sizes: &BTreeMap<DomainId, u64>) -> (u128, usize) {
    // FLOP proxy: product of sizes over the union of labels.
    let mut labels: Vec<u32> = a.labels.clone();
    for l in &b.labels {
        if !labels.contains(l) {
            labels.push(*l);
        }
    }
    let mut cost: u128 = 1;
    for l in &labels {
        let domain = a
            .domains
            .get(a.labels.iter().position(|x| x == l).unwrap_or(usize::MAX))
            .or_else(|| {
                b.domains
                    .get(b.labels.iter().position(|x| x == l).unwrap_or(usize::MAX))
            })
            .expect("label domain");
        cost *= sizes[domain] as u128;
    }
    let shared = a.labels.iter().filter(|l| b.labels.contains(l)).count();
    (cost, shared)
}

/// Greedily factor one n-ary einsum into a chain of binary einsums.
/// Returns the replacement definitions (intermediates then final op), where
/// intermediates are marked with negative placeholder refs resolved by caller.
fn factor_einsum(
    op: &Einsum,
    sizes: &BTreeMap<DomainId, u64>,
    next_fresh_value: &mut u32,
    emission: &mut Vec<ValueDef>,
) -> ValueDef {
    if op.args.len() <= 2 {
        return ValueDef::Einsum(op.clone());
    }
    let mut args: Vec<ArgView> = op
        .args
        .iter()
        .zip(&op.code.inputs)
        .map(|(r, labels)| ArgView {
            reference: *r,
            labels: labels.iter().map(|l| l.0).collect(),
            domains: labels.iter().map(|l| op.code.domains[l]).collect(),
        })
        .collect();
    let final_output: Vec<u32> = op.code.output.iter().map(|l| l.0).collect();

    while args.len() > 2 {
        // Pick the pair with minimum pairwise cost; tie-break on max shared.
        let mut best: Option<(usize, usize, u128, usize)> = None;
        for i in 0..args.len() {
            for j in (i + 1)..args.len() {
                let (cost, shared) = union_cost(&args[i], &args[j], sizes);
                let better = match best {
                    None => true,
                    Some((_, _, c, s)) => cost < c || (cost == c && shared > s),
                };
                if better {
                    best = Some((i, j, cost, shared));
                }
            }
        }
        let (i, j, _, _) = best.unwrap();
        let a = args[i].clone();
        let b = args[j].clone();
        // Remaining labels after removing i, j: labels needed later.
        let mut later: Vec<u32> = final_output.clone();
        for (k, arg) in args.iter().enumerate() {
            if k != i && k != j {
                for l in &arg.labels {
                    if !later.contains(l) {
                        later.push(*l);
                    }
                }
            }
        }
        // Intermediate labels: union of a, b, keeping only labels needed later.
        let mut inter_labels: Vec<u32> = Vec::new();
        let mut inter_domains: Vec<DomainId> = Vec::new();
        for arg in [&a, &b] {
            for (pos, l) in arg.labels.iter().enumerate() {
                if later.contains(l) && !inter_labels.contains(l) {
                    inter_labels.push(*l);
                    inter_domains.push(arg.domains[pos]);
                }
            }
        }
        let fresh_id = *next_fresh_value;
        *next_fresh_value += 1;
        // Build the intermediate's code with local IndexIds.
        let mut local: BTreeMap<u32, IndexId> = BTreeMap::new();
        let mut domains: BTreeMap<IndexId, DomainId> = BTreeMap::new();
        let mut next_local = 0u32;
        let mut localize = |l: u32, d: DomainId, local: &mut BTreeMap<u32, IndexId>,
                            domains: &mut BTreeMap<IndexId, DomainId>,
                            next_local: &mut u32| -> IndexId {
            if let Some(id) = local.get(&l) {
                return *id;
            }
            let id = IndexId(*next_local);
            *next_local += 1;
            local.insert(l, id);
            domains.insert(id, d);
            id
        };
        let in_a: Vec<IndexId> = a
            .labels
            .iter()
            .zip(&a.domains)
            .map(|(l, d)| localize(*l, *d, &mut local, &mut domains, &mut next_local))
            .collect();
        let in_b: Vec<IndexId> = b
            .labels
            .iter()
            .zip(&b.domains)
            .map(|(l, d)| localize(*l, *d, &mut local, &mut domains, &mut next_local))
            .collect();
        let out: Vec<IndexId> = inter_labels
            .iter()
            .zip(&inter_domains)
            .map(|(l, d)| localize(*l, *d, &mut local, &mut domains, &mut next_local))
            .collect();
        emission.push(ValueDef::Einsum(Einsum {
            coeff: Coefficient::from_integer(1.into()),
            code: Code {
                inputs: vec![in_a, in_b],
                output: out,
                domains,
            },
            args: vec![a.reference, b.reference],
        }));
        // Replace args i, j with the intermediate.
        args.remove(j);
        args.remove(i);
        args.push(ArgView {
            reference: ValueRef::Value(truthmill::ValueId(fresh_id)),
            labels: inter_labels,
            domains: inter_domains,
        });
    }
    // Final binary op: preserve the original output order and coefficient.
    let mut local: BTreeMap<u32, IndexId> = BTreeMap::new();
    let mut domains: BTreeMap<IndexId, DomainId> = BTreeMap::new();
    let mut next_local = 0u32;
    let mut localize = |l: u32, d: DomainId| -> IndexId {
        if let Some(id) = local.get(&l) {
            return *id;
        }
        let id = IndexId(next_local);
        next_local += 1;
        local.insert(l, id);
        domains.insert(id, d);
        id
    };
    let inputs: Vec<Vec<IndexId>> = args
        .iter()
        .map(|arg| {
            arg.labels
                .iter()
                .zip(&arg.domains)
                .map(|(l, d)| localize(*l, *d))
                .collect()
        })
        .collect();
    // Domains for output labels: look up from the original code.
    let output: Vec<IndexId> = op
        .code
        .output
        .iter()
        .map(|l| localize(l.0, op.code.domains[l]))
        .collect();
    ValueDef::Einsum(Einsum {
        coeff: op.coeff,
        code: Code {
            inputs,
            output,
            domains,
        },
        args: args.iter().map(|a| a.reference).collect(),
    })
}

/// Apply greedy factorization to every n-ary einsum in the program.
pub fn greedy_factor(program: &Program, sizes: &BTreeMap<DomainId, u64>) -> Program {
    factor_with_threshold(program, sizes, 0)
}

/// Factor only n-ary einsums whose union-index product exceeds
/// `threshold_bytes` (memory-bounded factoring): the backend handles small
/// n-ary ops well internally, but large ones can materialize huge internal
/// temporaries. Splitting them into greedy binary chains bounds every
/// intermediate by construction.
pub fn factor_with_threshold(
    program: &Program,
    sizes: &BTreeMap<DomainId, u64>,
    threshold_bytes: u128,
) -> Program {
    let mut values: Vec<ValueDef> = Vec::new();
    let mut remap: BTreeMap<u32, u32> = BTreeMap::new();
    for (old_id, definition) in program.values.iter().enumerate() {
        // Remap references in this definition.
        let remapped = remap_refs(definition, &remap);
        let new_def = match remapped {
            ValueDef::Einsum(op) => {
                let needs = op.args.len() > 2 && {
                    let mut labels: Vec<u32> = op.code.output.iter().map(|l| l.0).collect();
                    for input in &op.code.inputs {
                        for l in input {
                            if !labels.contains(&l.0) {
                                labels.push(l.0);
                            }
                        }
                    }
                    let union: u128 = labels
                        .iter()
                        .map(|l| sizes[&op.code.domains[&IndexId(*l)]] as u128)
                        .product::<u128>()
                        * 8;
                    union > threshold_bytes
                };
                if !needs {
                    remap.insert(old_id as u32, values.len() as u32);
                    values.push(ValueDef::Einsum(op));
                    continue;
                }
                // Intermediates occupy the next slots in `values`, so fresh
                // ids start at values.len().
                let mut emission = Vec::new();
                let mut next = values.len() as u32;
                let def = factor_einsum(&op, sizes, &mut next, &mut emission);
                values.extend(emission);
                remap.insert(old_id as u32, values.len() as u32);
                def
            }
            other => {
                remap.insert(old_id as u32, values.len() as u32);
                other
            }
        };
        values.push(new_def);
    }
    Program {
        values,
        outputs: program
            .outputs
            .iter()
            .map(|o| truthmill::ValueId(remap[&(o.0)]))
            .collect(),
    }
}

fn remap_refs(definition: &ValueDef, remap: &BTreeMap<u32, u32>) -> ValueDef {
    let fix = |r: &ValueRef| -> ValueRef {
        match r {
            ValueRef::Tensor(t) => ValueRef::Tensor(*t),
            ValueRef::Value(v) => ValueRef::Value(truthmill::ValueId(remap[&v.0])),
        }
    };
    match definition {
        ValueDef::Einsum(op) => ValueDef::Einsum(Einsum {
            coeff: op.coeff,
            code: op.code.clone(),
            args: op.args.iter().map(fix).collect(),
        }),
        ValueDef::LinComb(op) => ValueDef::LinComb(truthmill::LinComb {
            coeff: op.coeff.clone(),
            code: op.code.clone(),
            args: op.args.iter().map(fix).collect(),
        }),
    }
}
