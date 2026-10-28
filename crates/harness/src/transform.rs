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

/// Exact-duplicate CSE: collapse einsum definitions that are identical up to
/// label renaming (same args, same per-axis domains, same contraction
/// structure, same coefficient). truthmill's SSA treats duplicate definitions
/// as recomputation; this pass shares them. LinComb ops are left alone.
pub fn cse_exact(program: &Program) -> Program {
    use std::collections::HashMap;
    let mut values: Vec<ValueDef> = Vec::new();
    let mut remap: BTreeMap<u32, u32> = BTreeMap::new();
    let mut seen: HashMap<Vec<u8>, u32> = HashMap::new();

    fn signature(op: &Einsum) -> Vec<u8> {
        // Canonical relabeling: labels renamed 0,1,2.. by first appearance in
        // inputs-then-output order. Encodes args, domain sequences, structure,
        // and coefficient; invariant under per-code label renaming.
        let mut sig = Vec::new();
        let mut canon: BTreeMap<u32, u32> = BTreeMap::new();
        let mut next = 0u32;
        fn enc(
            sig: &mut Vec<u8>,
            l: u32,
            d: u32,
            canon: &mut BTreeMap<u32, u32>,
            next: &mut u32,
        ) {
            let c = *canon.entry(l).or_insert_with(|| {
                let v = *next;
                *next += 1;
                v
            });
            sig.extend_from_slice(&c.to_le_bytes());
            sig.extend_from_slice(&d.to_le_bytes());
        }
        for arg in &op.args {
            match arg {
                ValueRef::Tensor(t) => {
                    sig.push(0xFF);
                    sig.extend_from_slice(&t.0.to_le_bytes());
                }
                ValueRef::Value(v) => {
                    sig.push(0xFE);
                    sig.extend_from_slice(&v.0.to_le_bytes());
                }
            }
        }
        for labels in &op.code.inputs {
            sig.push(0xFD);
            for l in labels {
                enc(&mut sig, l.0, op.code.domains[l].0, &mut canon, &mut next);
            }
        }
        sig.push(0xFC);
        for l in &op.code.output {
            enc(&mut sig, l.0, op.code.domains[l].0, &mut canon, &mut next);
        }
        sig.push(0xFB);
        sig.extend_from_slice(&op.coeff.numer().to_le_bytes());
        sig.extend_from_slice(&op.coeff.denom().to_le_bytes());
        sig
    }

    for (old_id, definition) in program.values.iter().enumerate() {
        let remapped = remap_refs(definition, &remap);
        match remapped {
            ValueDef::Einsum(op) => {
                let sig = signature(&op);
                if let Some(&canonical) = seen.get(&sig) {
                    // Duplicate: alias, do not emit.
                    remap.insert(old_id as u32, canonical);
                } else {
                    let new_id = values.len() as u32;
                    seen.insert(sig, new_id);
                    remap.insert(old_id as u32, new_id);
                    values.push(ValueDef::Einsum(op));
                }
            }
            other => {
                remap.insert(old_id as u32, values.len() as u32);
                values.push(other);
            }
        }
    }
    Program {
        values,
        outputs: program
            .outputs
            .iter()
            .map(|o| truthmill::ValueId(remap[&o.0]))
            .collect(),
    }
}

/// Shared-pair intermediate discovery (cross-term factorization).
///
/// Finds pairs of arguments (A, B) in einsum ops whose pairwise contraction
/// recurs across ops (canonical up to label renaming), then introduces one
/// intermediate I = A . B and rewrites every occurrence to consume it.
/// Greedy: each round forms the group with the largest estimated saving
/// (count-1) * formation_flops; intermediates larger than
/// `max_intermediate_bytes` are skipped (memory guard). Verified downstream
/// by truthmill::verify + oracle.

use std::collections::HashMap;

#[derive(Clone)]
pub struct Occurrence {
    pub op: usize,
    pub arg_i: usize,
    pub arg_j: usize,
    /// canonical label id -> local IndexId in this op's code
    pub local_of_canon: BTreeMap<u32, IndexId>,
    /// free canonical ids in intermediate axis order (arg_i's free axes, then
    /// arg_j's new free axes)
    pub inter_order: Vec<u32>,
    pub formation_flops: f64,
}

fn canon_of(occ: &Occurrence, local: &IndexId) -> u32 {
    occ.local_of_canon
        .iter()
        .find(|(_, l)| *l == local)
        .map(|(c, _)| *c)
        .expect("label in canon map")
}

pub fn pair_occurrences(
    program: &Program,
    sizes: &BTreeMap<DomainId, u64>,
) -> HashMap<Vec<u8>, Vec<Occurrence>> {
    let mut groups: HashMap<Vec<u8>, Vec<Occurrence>> = HashMap::new();
    for (n, definition) in program.values.iter().enumerate() {
        let op = match definition {
            ValueDef::Einsum(op) => op,
            _ => continue,
        };
        let k = op.args.len();
        if k < 2 {
            continue;
        }
        for i in 0..k {
            for j in (i + 1)..k {
                let li = &op.code.inputs[i];
                let lj = &op.code.inputs[j];
                let contracted: Vec<IndexId> =
                    li.iter().filter(|l| lj.contains(l)).cloned().collect();
                if contracted.is_empty() {
                    continue;
                }
                let mut needed: Vec<IndexId> = op.code.output.clone();
                for (m, labels) in op.code.inputs.iter().enumerate() {
                    if m != i && m != j {
                        for l in labels {
                            if !needed.contains(l) {
                                needed.push(*l);
                            }
                        }
                    }
                }
                let pair_labels: Vec<IndexId> =
                    li.iter().chain(lj.iter()).cloned().collect();
                let free: Vec<IndexId> = pair_labels
                    .iter()
                    .filter(|l| needed.contains(l))
                    .cloned()
                    .collect();
                // Deterministic arg order by (ref, per-axis signature).
                let arg_key = |idx: usize| -> Vec<u8> {
                    let mut b = Vec::new();
                    match &op.args[idx] {
                        ValueRef::Tensor(t) => {
                            b.push(0xFF);
                            b.extend_from_slice(&t.0.to_le_bytes());
                        }
                        ValueRef::Value(v) => {
                            b.push(0xFE);
                            b.extend_from_slice(&v.0.to_le_bytes());
                        }
                    }
                    for l in &op.code.inputs[idx] {
                        b.extend_from_slice(&op.code.domains[l].0.to_le_bytes());
                        b.push(contracted.contains(l) as u8);
                        b.push(free.contains(l) as u8);
                    }
                    b
                };
                let (i, j) = if arg_key(i) <= arg_key(j) { (i, j) } else { (j, i) };
                let li = &op.code.inputs[i];
                let lj = &op.code.inputs[j];
                let mut canon: BTreeMap<u32, u32> = BTreeMap::new();
                let mut next = 0u32;
                let mut local_of_canon: BTreeMap<u32, IndexId> = BTreeMap::new();
                let mut sig = Vec::new();
                let mut inter_order: Vec<u32> = Vec::new();
                for labels in [li, lj] {
                    for l in labels {
                        let c = *canon.entry(l.0).or_insert_with(|| {
                            let v = next;
                            next += 1;
                            v
                        });
                        local_of_canon.insert(c, *l);
                        sig.extend_from_slice(&c.to_le_bytes());
                        sig.extend_from_slice(&op.code.domains[l].0.to_le_bytes());
                        sig.push(contracted.contains(l) as u8);
                        sig.push(free.contains(l) as u8);
                        if free.contains(l) {
                            inter_order.push(c);
                        }
                    }
                    sig.push(0xFD);
                }
                for idx in [i, j] {
                    match &op.args[idx] {
                        ValueRef::Tensor(t) => {
                            sig.push(0xFF);
                            sig.extend_from_slice(&t.0.to_le_bytes());
                        }
                        ValueRef::Value(v) => {
                            sig.push(0xFE);
                            sig.extend_from_slice(&v.0.to_le_bytes());
                        }
                    }
                }
                let union: Vec<IndexId> = {
                    let mut u = li.clone();
                    for l in lj {
                        if !u.contains(l) {
                            u.push(*l);
                        }
                    }
                    u
                };
                let formation: f64 = union
                    .iter()
                    .map(|l| sizes[&op.code.domains[l]] as f64)
                    .product();
                groups.entry(sig).or_default().push(Occurrence {
                    op: n,
                    arg_i: i,
                    arg_j: j,
                    local_of_canon,
                    inter_order,
                    formation_flops: formation,
                });
            }
        }
    }
    groups
}

/// One round: introduce the single most valuable shared intermediate.
fn share_one_round_auto(
    program: &Program,
    sizes: &BTreeMap<DomainId, u64>,
    max_intermediate_bytes: usize,
) -> Option<Program> {
    let groups = pair_occurrences(program, sizes);
    let mut best: Option<(f64, &Vec<Occurrence>)> = None;
    for occs in groups.values() {
        if occs.len() < 2 {
            continue;
        }
        let first = &occs[0];
        let op = match &program.values[first.op] {
            ValueDef::Einsum(op) => op,
            _ => unreachable!(),
        };
        let inter_elems: usize = first
            .inter_order
            .iter()
            .map(|c| sizes[&op.code.domains[&first.local_of_canon[c]]] as usize)
            .product();
        if inter_elems * 8 > max_intermediate_bytes {
            continue;
        }
        let saving = (occs.len() - 1) as f64 * first.formation_flops;
        if best.as_ref().map(|(s, _)| saving > *s).unwrap_or(true) {
            best = Some((saving, occs));
        }
    }
    let (_, occs) = best?;
    Some(share_introduce(program, occs))
}

/// Introduce the shared intermediate for a specific occurrence group.
pub fn share_introduce(program: &Program, occs: &[Occurrence]) -> Program {
    let leader_pos = occs.iter().map(|o| o.op).min().unwrap();
    let leader = occs.iter().find(|o| o.op == leader_pos).unwrap().clone();
    let leader_op = match &program.values[leader_pos] {
        ValueDef::Einsum(op) => op,
        _ => unreachable!(),
    };

    // Build the intermediate definition (local labels in canon order).
    let mut domains: BTreeMap<IndexId, DomainId> = BTreeMap::new();
    let mut next_local = 0u32;
    let mut canon_to_local: BTreeMap<u32, IndexId> = BTreeMap::new();
    let mut localize = |canon_id: u32, domain: DomainId| {
        *canon_to_local.entry(canon_id).or_insert_with(|| {
            let id = IndexId(next_local);
            next_local += 1;
            domains.insert(id, domain);
            id
        })
    };
    let canon_of_local = |l: &IndexId| -> u32 {
        leader
            .local_of_canon
            .iter()
            .find(|(_, x)| *x == l)
            .map(|(c, _)| *c)
            .expect("leader canon")
    };
    let in_a: Vec<IndexId> = leader_op.code.inputs[leader.arg_i]
        .iter()
        .map(|l| {
            let c = canon_of_local(l);
            localize(c, leader_op.code.domains[l])
        })
        .collect();
    let in_b: Vec<IndexId> = leader_op.code.inputs[leader.arg_j]
        .iter()
        .map(|l| {
            let c = canon_of_local(l);
            localize(c, leader_op.code.domains[l])
        })
        .collect();
    let inter_out: Vec<IndexId> = leader
        .inter_order
        .iter()
        .map(|c| localize(*c, leader_op.code.domains[&leader.local_of_canon[c]]))
        .collect();
    let inter_def = ValueDef::Einsum(Einsum {
        coeff: Coefficient::from_integer(1.into()),
        code: Code {
            inputs: vec![in_a, in_b],
            output: inter_out,
            domains,
        },
        args: vec![
            leader_op.args[leader.arg_i],
            leader_op.args[leader.arg_j],
        ],
    });

    // Rewrite: insert inter_def before leader_pos; shift ids; rewrite occs.
    let occ_ops: std::collections::BTreeSet<usize> = occs.iter().map(|o| o.op).collect();
    let shift = |v: u32| -> u32 {
        if v as usize >= leader_pos {
            v + 1
        } else {
            v
        }
    };
    let shift_ref = |r: &ValueRef| -> ValueRef {
        match r {
            ValueRef::Tensor(t) => ValueRef::Tensor(*t),
            ValueRef::Value(v) => ValueRef::Value(truthmill::ValueId(shift(v.0))),
        }
    };
    let mut values: Vec<ValueDef> = Vec::with_capacity(program.values.len() + 1);
    for (n, definition) in program.values.iter().enumerate() {
        if n == leader_pos {
            values.push(inter_def.clone());
        }
        if occ_ops.contains(&n) {
            let op = match definition {
                ValueDef::Einsum(op) => op,
                _ => unreachable!(),
            };
            let occ = occs.iter().find(|o| o.op == n).unwrap();
            let (i, j) = (occ.arg_i, occ.arg_j);
            let mut inputs: Vec<Vec<IndexId>> = Vec::new();
            let mut args: Vec<ValueRef> = Vec::new();
            for m in 0..op.args.len() {
                if m == i || m == j {
                    continue;
                }
                inputs.push(op.code.inputs[m].clone());
                args.push(shift_ref(&op.args[m]));
            }
            let inter_labels: Vec<IndexId> = occ
                .inter_order
                .iter()
                .map(|c| occ.local_of_canon[c])
                .collect();
            inputs.push(inter_labels);
            args.push(ValueRef::Value(truthmill::ValueId(leader_pos as u32)));
            // domains must cover exactly the used labels
            let mut domains = BTreeMap::new();
            for labels in inputs.iter().chain(std::iter::once(&op.code.output)) {
                for l in labels {
                    domains.insert(*l, op.code.domains[l]);
                }
            }
            values.push(ValueDef::Einsum(Einsum {
                coeff: op.coeff,
                code: Code {
                    inputs,
                    output: op.code.output.clone(),
                    domains,
                },
                args,
            }));
        } else {
            let mut def = definition.clone();
            match &mut def {
                ValueDef::Einsum(op) => {
                    op.args = op.args.iter().map(shift_ref).collect();
                }
                ValueDef::LinComb(op) => {
                    op.args = op.args.iter().map(shift_ref).collect();
                }
            }
            values.push(def);
        }
    }
    Program {
        values,
        outputs: program
            .outputs
            .iter()
            .map(|o| truthmill::ValueId(shift(o.0)))
            .collect(),
    }
}

/// Iterate shared-pair discovery until no profitable group remains.
pub fn share_pairs(
    program: &Program,
    sizes: &BTreeMap<DomainId, u64>,
    max_rounds: usize,
    max_intermediate_bytes: usize,
) -> Program {
    let mut current = program.clone();
    for _ in 0..max_rounds {
        match share_one_round_auto(&current, sizes, max_intermediate_bytes) {
            Some(next) => current = next,
            None => break,
        }
    }
    current
}
