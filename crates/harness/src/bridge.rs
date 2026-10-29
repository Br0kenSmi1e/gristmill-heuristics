//! Bridge between truthmill `Problem`/`Program` (execution side) and
//! gristmill_rewrites `Computation`/`State` (rewrite side).
//!
//! truthmill -> rewrites: expand each output of the reference SSA program
//! into a flat sum-of-products definition (polynomial substitution).
//! rewrites -> truthmill: each definition's terms become Einsum values
//! summed by a LinComb; definitions referencing earlier definitions become
//! SSA value references.

use std::collections::{BTreeMap, BTreeSet};

use gristmill_rewrites::repr::{
    Computation, Index as RIndex, RangeId, SymmetryAction, TensorDef as RTensorDef,
    TensorInfo, TensorRef as RTensorRef, Term as RTerm,
};
use truthmill::{
    Code, DomainId, Einsum, IndexId, LinComb, Problem, Program, TensorId, ValueDef, ValueId,
    ValueRef,
};

// ---------------------------------------------------------------------------
// truthmill -> rewrites: expansion
// ---------------------------------------------------------------------------

/// A global expanded index: identity plus domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct GIndex {
    id: u64,
    domain: DomainId,
}

#[derive(Clone)]
struct PolyTerm {
    coeff: truthmill::Coefficient,
    factors: Vec<(TensorId, Vec<GIndex>)>,
    sums: Vec<GIndex>, // bound indices
}

#[derive(Clone)]
struct Poly {
    free: Vec<GIndex>, // external indices in axis order
    terms: Vec<PolyTerm>,
}

struct Expander {
    next_global: u64,
}

impl Expander {
    fn fresh(&mut self, domain: DomainId) -> GIndex {
        let id = self.next_global;
        self.next_global += 1;
        GIndex { id, domain }
    }

    fn expand_program(&mut self, program: &Program) -> Vec<Poly> {
        let mut polys: Vec<Poly> = Vec::new();
        for definition in &program.values {
            match definition {
                ValueDef::Einsum(op) => {
                    // Fresh globals for every code-local label.
                    let globals: BTreeMap<u32, GIndex> = op
                        .code
                        .domains
                        .iter()
                        .map(|(l, d)| (l.0, self.fresh(*d)))
                        .collect();
                    let out_globals: Vec<GIndex> =
                        op.code.output.iter().map(|l| globals[&l.0]).collect();
                    // Multiply argument polynomials, renaming each occurrence.
                    let mut terms = vec![PolyTerm {
                        coeff: op.coeff,
                        factors: Vec::new(),
                        sums: Vec::new(),
                    }];
                    for (arg, labels) in op.args.iter().zip(&op.code.inputs) {
                        match arg {
                            ValueRef::Tensor(t) => {
                                let idxs: Vec<GIndex> =
                                    labels.iter().map(|l| globals[&l.0]).collect();
                                for t0 in &mut terms {
                                    t0.factors.push((*t, idxs.clone()));
                                }
                            }
                            ValueRef::Value(v) => {
                                let sub = polys[v.0 as usize].clone();
                                let mut renamed: Vec<PolyTerm> = Vec::new();
                                for t0 in &terms {
                                    for st in &sub.terms {
                                        // Per-occurrence rename: summed ->
                                        // fresh globals; free at axis pos ->
                                        // this code's labels[pos] global.
                                        let mut ren: BTreeMap<u64, GIndex> =
                                            BTreeMap::new();
                                        for gi in &st.sums {
                                            ren.insert(
                                                gi.id,
                                                self.fresh(gi.domain),
                                            );
                                        }
                                        for (pos, gi) in sub.free.iter().enumerate() {
                                            ren.insert(gi.id, globals[&labels[pos].0]);
                                        }
                                        let mut nt = PolyTerm {
                                            coeff: t0.coeff * st.coeff,
                                            factors: t0.factors.clone(),
                                            sums: t0.sums.clone(),
                                        };
                                        nt.sums.extend(
                                            st.sums.iter().map(|gi| ren[&gi.id]),
                                        );
                                        for (tensor, idxs) in &st.factors {
                                            nt.factors.push((
                                                *tensor,
                                                idxs.iter().map(|gi| ren[&gi.id]).collect(),
                                            ));
                                        }
                                        renamed.push(nt);
                                    }
                                }
                                terms = renamed;
                            }
                        }
                    }
                    // Summed = used but not free.
                    for t in &mut terms {
                        let mut used: Vec<GIndex> = Vec::new();
                        for (_, idxs) in &t.factors {
                            for gi in idxs {
                                if !used.contains(gi) {
                                    used.push(*gi);
                                }
                            }
                        }
                        let mut sums: Vec<GIndex> = used
                            .into_iter()
                            .filter(|gi| !out_globals.contains(gi))
                            .collect();
                        sums.sort();
                        t.sums = sums;
                    }
                    polys.push(Poly {
                        free: out_globals,
                        terms,
                    });
                }
                ValueDef::LinComb(op) => {
                    let globals: BTreeMap<u32, GIndex> = op
                        .code
                        .domains
                        .iter()
                        .map(|(l, d)| (l.0, self.fresh(*d)))
                        .collect();
                    let out_globals: Vec<GIndex> =
                        op.code.output.iter().map(|l| globals[&l.0]).collect();
                    let mut terms = Vec::new();
                    for (coeff, (arg, labels)) in
                        op.coeff.iter().zip(op.args.iter().zip(&op.code.inputs))
                    {
                        match arg {
                            ValueRef::Tensor(t) => {
                                terms.push(PolyTerm {
                                    coeff: *coeff,
                                    factors: vec![(
                                        *t,
                                        labels.iter().map(|l| globals[&l.0]).collect(),
                                    )],
                                    sums: Vec::new(),
                                });
                            }
                            ValueRef::Value(v) => {
                                let sub = polys[v.0 as usize].clone();
                                for st in &sub.terms {
                                    let mut ren: BTreeMap<u64, GIndex> = BTreeMap::new();
                                    for gi in &st.sums {
                                        ren.insert(gi.id, self.fresh(gi.domain));
                                    }
                                    for (pos, gi) in sub.free.iter().enumerate() {
                                        ren.insert(gi.id, globals[&labels[pos].0]);
                                    }
                                    terms.push(PolyTerm {
                                        coeff: *coeff * st.coeff,
                                        factors: st
                                            .factors
                                            .iter()
                                            .map(|(tensor, idxs)| {
                                                (
                                                    *tensor,
                                                    idxs.iter()
                                                        .map(|gi| ren[&gi.id])
                                                        .collect(),
                                                )
                                            })
                                            .collect(),
                                        sums: st.sums.iter().map(|gi| ren[&gi.id]).collect(),
                                    });
                                }
                            }
                        }
                    }
                    polys.push(Poly {
                        free: out_globals,
                        terms,
                    });
                }
            }
        }
        polys
    }
}

/// Convert a truthmill Problem into a rewrites Computation: one definition
/// per output, fresh output TensorIds from `output_base` upward.
pub fn problem_to_computation(problem: &Problem, output_base: u32) -> (Computation, Vec<gristmill_rewrites::repr::TensorId>) {
    let mut expander = Expander { next_global: 1 };
    let polys = expander.expand_program(&problem.reference);

    let mut tensors: BTreeMap<gristmill_rewrites::repr::TensorId, TensorInfo> = BTreeMap::new();
    // Input tensors: rank from symmetries or first use.
    let patterns = crate::dense::collect_patterns(problem);
    for (tensor, pats) in &patterns {
        let rank = pats[0].len();
        let symmetry = problem
            .symmetries
            .get(tensor)
            .map(|gens| {
                gens.iter()
                    .map(|g| gristmill_rewrites::repr::SymmetryGenerator {
                        perm: g.permutation.clone(),
                        action: if g.sign == -1 {
                            SymmetryAction::Negate
                        } else {
                            SymmetryAction::Identity
                        },
                    })
                    .collect()
            })
            .unwrap_or_default();
        tensors.insert(
            gristmill_rewrites::repr::TensorId(tensor.0),
            TensorInfo { rank, symmetry },
        );
    }

    let mut definitions = Vec::new();
    let mut outputs = Vec::new();
    for (k, o) in problem.reference.outputs.iter().enumerate() {
        let poly = &polys[o.0 as usize];
        let out_tid = gristmill_rewrites::repr::TensorId(output_base + k as u32);
        // External indices: kernel requires definition-local IndexIds; use the
        // poly's global ids directly as index identities (unique per index).
        let exts: Vec<RIndex> = poly
            .free
            .iter()
            .map(|gi| RIndex {
                id: gristmill_rewrites::repr::IndexId(gi.id as u32),
                range: RangeId(gi.domain.0),
            })
            .collect();
        let rhs: Vec<RTerm> = poly
            .terms
            .iter()
            .map(|t| RTerm {
                sums: t
                    .sums
                    .iter()
                    .map(|gi| RIndex {
                        id: gristmill_rewrites::repr::IndexId(gi.id as u32),
                        range: RangeId(gi.domain.0),
                    })
                    .collect(),
                coeff: num_rational::BigRational::new(
                    (*t.coeff.numer()).into(),
                    (*t.coeff.denom()).into(),
                ),
                factors: t
                    .factors
                    .iter()
                    .map(|(tensor, idxs)| RTensorRef {
                        tensor: gristmill_rewrites::repr::TensorId(tensor.0),
                        indices: idxs
                            .iter()
                            .map(|gi| gristmill_rewrites::repr::IndexId(gi.id as u32))
                            .collect(),
                    })
                    .collect(),
            })
            .collect();
        tensors.insert(
            out_tid,
            TensorInfo {
                rank: exts.len(),
                symmetry: Vec::new(),
            },
        );
        definitions.push(RTensorDef {
            base: out_tid,
            exts,
            rhs,
        });
        outputs.push(out_tid);
    }

    let range_set: BTreeSet<RangeId> =
        problem.sizes.keys().map(|d| RangeId(d.0)).collect();
    (
        Computation {
            ranges: range_set,
            tensors,
            definitions,
        },
        outputs,
    )
}

// ---------------------------------------------------------------------------
// rewrites -> truthmill: candidate program
// ---------------------------------------------------------------------------

/// Convert a (possibly rewritten) Computation back into a truthmill Program.
/// `input_tensors` are the original problem's tensor ids; any other tensor is
/// an intermediate defined by an earlier definition. `outputs` are the
/// protected output tensor ids, in the reference program's output order.
pub fn computation_to_program(
    computation: &Computation,
    input_tensors: &[u32],
    outputs: &[u32],
) -> Program {
    let mut values: Vec<ValueDef> = Vec::new();
    let mut def_value: BTreeMap<u32, u32> = BTreeMap::new(); // TensorId -> ValueId

    // Emit definitions in dependency order (kernel vector order is not
    // guaranteed topological): fixpoint over known references.
    let mut remaining: Vec<&gristmill_rewrites::repr::TensorDef> =
        computation.definitions.iter().collect();
    while !remaining.is_empty() {
        let before = remaining.len();
        remaining.retain(|def| {
            let ready = def.rhs.iter().all(|t| {
                t.factors.iter().all(|f| {
                    input_tensors.contains(&f.tensor.0) || def_value.contains_key(&f.tensor.0)
                })
            });
            !ready
        });
        if remaining.len() == before {
            anyhow_panic("cyclic or unresolvable definition references");
        }
        let ready_now: Vec<_> = computation
            .definitions
            .iter()
            .filter(|def| {
                def.rhs.iter().all(|t| {
                    t.factors.iter().all(|f| {
                        input_tensors.contains(&f.tensor.0) || def_value.contains_key(&f.tensor.0)
                    })
                }) && !def_value.contains_key(&def.base.0)
            })
            .collect();
        for def in ready_now {
            emit_definition(def, &input_tensors, &mut values, &mut def_value);
        }
    }
    // (loop replaces the old straight-line iteration)

    return Program {
        values,
        outputs: outputs.iter().map(|o| ValueId(def_value[o])).collect(),
    };
}

#[track_caller]
fn anyhow_panic(msg: &str) -> ! {
    panic!("computation_to_program: {msg}")
}

fn emit_definition(
    def: &gristmill_rewrites::repr::TensorDef,
    input_tensors: &[u32],
    values: &mut Vec<ValueDef>,
    def_value: &mut BTreeMap<u32, u32>,
) {
        let base = def.base.0;
        let mut term_values = Vec::new();
        for term in &def.rhs {
            // Code-local label numbering.
            let mut local: BTreeMap<u32, IndexId> = BTreeMap::new();
            let mut domains: BTreeMap<IndexId, DomainId> = BTreeMap::new();
            let mut localize = |gid: u32, dom: u32| -> IndexId {
                if let Some(id) = local.get(&gid) {
                    return *id;
                }
                let id = IndexId(local.len() as u32);
                local.insert(gid, id);
                domains.insert(id, DomainId(dom));
                id
            };
            let sum_domains: BTreeMap<u32, u32> =
                term.sums.iter().map(|s| (s.id.0, s.range.0)).collect();
            let ext_domains: BTreeMap<u32, u32> =
                def.exts.iter().map(|e| (e.id.0, e.range.0)).collect();
            let inputs: Vec<Vec<IndexId>> = term
                .factors
                .iter()
                .map(|f| {
                    f.indices
                        .iter()
                        .map(|i| {
                            let dom = ext_domains
                                .get(&i.0)
                                .or_else(|| sum_domains.get(&i.0))
                                .copied()
                                .expect("index domain");
                            localize(i.0, dom)
                        })
                        .collect()
                })
                .collect();
            let output: Vec<IndexId> = def
                .exts
                .iter()
                .map(|e| localize(e.id.0, e.range.0))
                .collect();
            let args: Vec<ValueRef> = term
                .factors
                .iter()
                .map(|f| {
                    if input_tensors.contains(&f.tensor.0) {
                        ValueRef::Tensor(TensorId(f.tensor.0))
                    } else {
                        ValueRef::Value(ValueId(def_value[&f.tensor.0]))
                    }
                })
                .collect();
            let coeff = term.coeff.clone();
            let numer: i64 = coeff.numer().try_into().expect("coeff numer fits i64");
            let denom: i64 = coeff.denom().try_into().expect("coeff denom fits i64");
            values.push(ValueDef::Einsum(Einsum {
                coeff: truthmill::Coefficient::new(numer, denom),
                code: Code {
                    inputs,
                    output,
                    domains,
                },
                args,
            }));
            term_values.push(values.len() as u32 - 1);
        }
        let final_value = if term_values.len() == 1 {
            term_values[0]
        } else {
            // LinComb sum, identity permutations.
            let rank = def.exts.len();
            let domains: BTreeMap<IndexId, DomainId> = def
                .exts
                .iter()
                .enumerate()
                .map(|(k, e)| (IndexId(k as u32), DomainId(e.range.0)))
                .collect();
            let inputs = vec![(0..rank as u32).map(IndexId).collect::<Vec<_>>(); term_values.len()];
            values.push(ValueDef::LinComb(LinComb {
                coeff: vec![truthmill::Coefficient::new(1, 1); term_values.len()],
                code: Code {
                    inputs,
                    output: (0..rank as u32).map(IndexId).collect(),
                    domains,
                },
                args: term_values.iter().map(|v| ValueRef::Value(ValueId(*v))).collect(),
            }));
            values.len() as u32 - 1
        };
        def_value.insert(base, final_value);
}

/// Split definitions with more than `max_terms` terms into chunk
/// sub-definitions plus a final sum definition (selector-side budget for
/// exponential biclique enumeration). Semantics preserved by construction;
/// cross-chunk factorizations are lost (documented tradeoff).
fn flush_chunk(
    comp: &mut Computation,
    new_defs: &mut Vec<RTensorDef>,
    chunk: &mut Vec<RTerm>,
    part_ids: &mut Vec<u32>,
    next_id: &mut u32,
    exts: &[RIndex],
) {
    if chunk.is_empty() {
        return;
    }
    let pid = *next_id;
    *next_id += 1;
    comp.tensors.insert(
        gristmill_rewrites::repr::TensorId(pid),
        TensorInfo {
            rank: exts.len(),
            symmetry: Vec::new(),
        },
    );
    new_defs.push(RTensorDef {
        base: gristmill_rewrites::repr::TensorId(pid),
        exts: exts.to_vec(),
        rhs: std::mem::take(chunk),
    });
    part_ids.push(pid);
}

pub fn split_large_definitions(comp: &mut Computation, max_terms: usize) {
    let mut next_id = comp.tensors.keys().map(|t| t.0).max().unwrap_or(0) + 1;
    let mut new_defs: Vec<RTensorDef> = Vec::new();
    for def in std::mem::take(&mut comp.definitions) {
        if def.rhs.len() <= max_terms {
            new_defs.push(def);
            continue;
        }
        let exts = def.exts.clone();
        let mut part_ids = Vec::new();
        let mut chunk: Vec<RTerm> = Vec::new();
        for term in def.rhs {
            chunk.push(term);
            if chunk.len() >= max_terms {
                flush_chunk(comp, &mut new_defs, &mut chunk, &mut part_ids, &mut next_id, &exts);
            }
        }
        flush_chunk(comp, &mut new_defs, &mut chunk, &mut part_ids, &mut next_id, &exts);
        // Final sum definition with the original base id.
        let rhs = part_ids
            .iter()
            .map(|pid| RTerm {
                sums: Vec::new(),
                coeff: num_rational::BigRational::from_integer(1.into()),
                factors: vec![RTensorRef {
                    tensor: gristmill_rewrites::repr::TensorId(*pid),
                    indices: exts.iter().map(|e| e.id).collect(),
                }],
            })
            .collect();
        new_defs.push(RTensorDef {
            base: def.base,
            exts,
            rhs,
        });
    }
    comp.definitions = new_defs;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_mp3_bridge_roundtrip_matches_scalar_oracle() {
        let mut problem = Problem::from_json(include_str!(
            "../../../benches/chemistry/mp3.json"
        ))
        .unwrap();
        problem.sizes.insert(DomainId(0), 3);
        problem.sizes.insert(DomainId(1), 5);
        let input_ids: Vec<u32> = crate::dense::collect_patterns(&problem)
            .keys()
            .map(|t| t.0)
            .collect();
        let (computation, outputs) = problem_to_computation(&problem, 1000);
        let output_ids: Vec<u32> = outputs.iter().map(|t| t.0).collect();
        let program = computation_to_program(&computation, &input_ids, &output_ids);
        assert!(truthmill::verify(&problem, &program).unwrap());
        let blocks = crate::dense::generate_inputs_for(&problem, &[&program], 0);
        let expected = crate::oracle::eval_program(&problem, &problem.reference, &blocks).unwrap();
        let actual = crate::oracle::eval_program(&problem, &program, &blocks).unwrap();
        for (expected, actual) in expected.iter().zip(&actual) {
            let max_diff = expected
                .data
                .iter()
                .zip(&actual.data)
                .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
            assert!(max_diff < 1e-12, "bridge roundtrip diff {max_diff}");
        }
    }

    #[test]
    fn canonicalized_mp3_bridge_roundtrip_matches_scalar_oracle() {
        let mut problem = Problem::from_json(include_str!(
            "../../../benches/chemistry/mp3.json"
        ))
        .unwrap();
        problem.sizes.insert(DomainId(0), 3);
        problem.sizes.insert(DomainId(1), 5);
        let input_ids: Vec<u32> = crate::dense::collect_patterns(&problem)
            .keys()
            .map(|t| t.0)
            .collect();
        let (computation, outputs) = problem_to_computation(&problem, 1000);
        let output_ids: Vec<u32> = outputs.iter().map(|t| t.0).collect();
        let state = gristmill_rewrites::state::State::new(computation, outputs).unwrap();
        let program = computation_to_program(state.computation(), &input_ids, &output_ids);
        assert!(truthmill::verify(&problem, &program).unwrap());
        let blocks = crate::dense::generate_inputs_for(&problem, &[&program], 0);
        let expected = crate::oracle::eval_program(&problem, &problem.reference, &blocks).unwrap();
        let actual = crate::oracle::eval_program(&problem, &program, &blocks).unwrap();
        for (expected, actual) in expected.iter().zip(&actual) {
            let max_diff = expected
                .data
                .iter()
                .zip(&actual.data)
                .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
            assert!(max_diff < 1e-12, "canonicalized bridge diff {max_diff}");
        }
    }
}
