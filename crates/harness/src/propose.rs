//! Structural proposer: direct intermediate proposals from term structure,
//! near-linear in equation size (no biclique enumeration).
//!
//! Signature of a pair of factor occurrences within a term: the two tensor
//! ids (sorted), per-tensor sorted (domain, paired-with-other?) axis lists,
//! and the sorted multiset of free (unpaired) domains. Terms sharing a
//! signature can share the corresponding pairwise-contraction intermediate up
//! to axis permutation. Grouping is O(terms x factors^2) hashing; proposals
//! are constructed directly as new Computations and gated later by exact
//! verification (kernel equivalent_states + truthmill::verify).

use std::collections::BTreeMap;

use gristmill_rewrites::repr::{
    Computation, Index as RIndex, RangeId, TensorDef as RTensorDef, TensorId, TensorInfo,
    TensorRef as RTensorRef, Term as RTerm,
};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(crate) struct PairSig {
    ta: u32,
    tb: u32,
    // Per-axis: (domain, partner position in the other tensor or usize::MAX).
    axes_a: Vec<(u32, usize)>,
    axes_b: Vec<(u32, usize)>,
    free_domains: Vec<u32>,
}

/// One occurrence of a factor pair: definition index, term index, the two
/// factor positions (ordered to match sig.ta/tb), and the axis index ids.
#[derive(Clone)]
pub(crate) struct PairOcc {
    def: usize,
    term: usize,
    fa: usize,
    fb: usize,
    axes_a: Vec<u32>,
    axes_b: Vec<u32>,
    free_exts: Vec<u32>, // ids of free axes (a then b order)
}

pub(crate) fn pair_signature(
    comp: &Computation,
    def_i: usize,
    term_i: usize,
    fa: usize,
    fb: usize,
) -> Option<(PairSig, PairOcc)> {
    let def = &comp.definitions[def_i];
    let term = &def.rhs[term_i];
    let f1 = &term.factors[fa];
    let f2 = &term.factors[fb];
    let (t1, t2) = (f1.tensor.0, f2.tensor.0);
    // Domain lookup: external ids from def.exts, summed from term.sums.
    let dom = |id: u32| -> u32 {
        def.exts
            .iter()
            .find(|e| e.id.0 == id)
            .map(|e| e.range.0)
            .or_else(|| term.sums.iter().find(|s| s.id.0 == id).map(|s| s.range.0))
            .expect("index domain")
    };
    // paired = index id occurs in the other factor; record partner position.
    let axes1: Vec<(u32, usize)> = f1
        .indices
        .iter()
        .map(|i| {
            (
                dom(i.0),
                f2.indices.iter().position(|j| j == i).unwrap_or(usize::MAX),
            )
        })
        .collect();
    let axes2: Vec<(u32, usize)> = f2
        .indices
        .iter()
        .map(|i| {
            (
                dom(i.0),
                f1.indices.iter().position(|j| j == i).unwrap_or(usize::MAX),
            )
        })
        .collect();
    // Skip if a paired index appears in a third factor (not a pairwise
    // contraction) or if a factor has repeated (diagonal) indices.
    let mut count: BTreeMap<u32, usize> = BTreeMap::new();
    for f in &term.factors {
        for i in &f.indices {
            *count.entry(i.0).or_default() += 1;
        }
    }
    for i in f1.indices.iter().chain(f2.indices.iter()) {
        let in_both = f1.indices.contains(i) && f2.indices.contains(i);
        if in_both {
            if count[&i.0] != 2 {
                return None;
            }
        }
    }
    for f in [f1, f2] {
        let mut seen = std::collections::BTreeSet::new();
        for i in &f.indices {
            if !seen.insert(i.0) {
                return None;
            }
        }
    }
    let (ta, tb, axes_a, axes_b, fa2, fb2) = if t1 <= t2 {
        (t1, t2, axes1, axes2, fa, fb)
    } else {
        (t2, t1, axes2, axes1, fb, fa)
    };
    let mut sa = axes_a.clone();
    sa.sort();
    let mut sb = axes_b.clone();
    sb.sort();
    let free_domains: Vec<u32> = axes_a
        .iter()
        .chain(axes_b.iter())
        .filter(|(_, partner)| *partner == usize::MAX)
        .map(|(d, _)| *d)
        .collect::<Vec<_>>()
        .tap_sort();
    let sig = PairSig {
        ta,
        tb,
        axes_a: sa,
        axes_b: sb,
        free_domains,
    };
    let occ = PairOcc {
        def: def_i,
        term: term_i,
        fa: fa2,
        fb: fb2,
        axes_a: f1.indices.iter().map(|i| i.0).collect(),
        axes_b: f2.indices.iter().map(|i| i.0).collect(),
        free_exts: Vec::new(),
    };
    let _ = comp;
    Some((sig, occ))
}

trait TapSort {
    fn tap_sort(self) -> Self;
}
impl TapSort for Vec<u32> {
    fn tap_sort(mut self) -> Self {
        self.sort();
        self
    }
}

/// A proposal: form intermediate I = contraction of the pair, rewrite all
/// member terms to use it.
pub struct Proposal {
    pub group_size: usize,
    pub free_domains: Vec<u32>,
    pub est_saving_terms: usize,
    occs: Vec<PairOcc>,
    ta: u32,
    tb: u32,
}

/// Group all pair occurrences by structural signature.
pub fn propose(comp: &Computation) -> Vec<Proposal> {
    let mut groups: BTreeMap<PairSig, Vec<PairOcc>> = BTreeMap::new();
    for (d, def) in comp.definitions.iter().enumerate() {
        for (t, term) in def.rhs.iter().enumerate() {
            let nf = term.factors.len();
            for fa in 0..nf {
                for fb in (fa + 1)..nf {
                    if let Some((sig, occ)) = pair_signature(comp, d, t, fa, fb) {
                        groups.entry(sig).or_default().push(occ);
                    }
                }
            }
        }
    }
    groups
        .into_iter()
        .map(|(sig, occs)| {
            // One occurrence per term: a second occurrence's factor positions
            // would be invalidated by the first rewrite. Remaining overlap
            // becomes next-step proposals.
            let mut seen = std::collections::BTreeSet::new();
            let occs: Vec<PairOcc> = occs
                .into_iter()
                .filter(|o| seen.insert((o.def, o.term)))
                .collect();
            (sig, occs)
        })
        .filter(|(_, occs)| occs.len() >= 2)
        .map(|(sig, occs)| {
            let n = occs.len();
            Proposal {
                group_size: n,
                est_saving_terms: n,
                free_domains: sig.free_domains.clone(),
                occs,
                ta: sig.ta,
                tb: sig.tb,
            }
        })
        .collect()
}

/// Materialize a proposal: returns a new Computation with the intermediate
/// definition inserted before the earliest referencing definition, and all
/// member terms rewritten to reference it.
pub fn apply_proposal(comp: &Computation, proposal: &Proposal) -> Computation {
    let dbg = std::env::var("H_DEBUG").is_ok();
    if dbg { eprintln!("apply: start, group {}", proposal.group_size); }
    let mut out = comp.clone();
    if dbg { eprintln!("apply: cloned"); }
    let new_tid = out.tensors.keys().map(|t| t.0).max().unwrap_or(0) + 1;
    // Canonical intermediate axes: free axes of A (in first occurrence's A
    // axis order) then free axes of B.
    let first = &proposal.occs[0];
    let first_term = &comp.definitions[first.def].rhs[first.term];
    let fa = &first_term.factors[first.fa];
    let fb = &first_term.factors[first.fb];
    let paired: std::collections::BTreeSet<u32> = fa
        .indices
        .iter()
        .filter(|i| fb.indices.contains(i))
        .map(|i| i.0)
        .collect();
    let dom = |def: &RTensorDef, term: &RTerm, id: u32| -> u32 {
        def.exts
            .iter()
            .find(|e| e.id.0 == id)
            .map(|e| e.range.0)
            .or_else(|| term.sums.iter().find(|s| s.id.0 == id).map(|s| s.range.0))
            .expect("domain")
    };
    // Canonical free-axis order: sorted by (domain, position) so every
    // occurrence agrees on the role of each canonical axis.
    let mut free_a: Vec<(usize, u32)> = fa
        .indices
        .iter()
        .enumerate()
        .filter(|(_, i)| !paired.contains(&i.0))
        .map(|(p, i)| (p, i.0))
        .collect();
    free_a.sort_by_key(|(p, i)| (dom(&comp.definitions[first.def], &comp.definitions[first.def].rhs[first.term], *i), *p));
    let mut free_b: Vec<(usize, u32)> = fb
        .indices
        .iter()
        .enumerate()
        .filter(|(_, i)| !paired.contains(&i.0))
        .map(|(p, i)| (p, i.0))
        .collect();
    free_b.sort_by_key(|(p, i)| (dom(&comp.definitions[first.def], &comp.definitions[first.def].rhs[first.term], *i), *p));
    // Fresh ids for the intermediate definition.
    let mut next_id: u32 = 1_000_000;
    let mut fresh = |next_id: &mut u32| {
        let id = *next_id;
        *next_id += 1;
        gristmill_rewrites::repr::IndexId(id)
    };
    let mut ext_map: Vec<(u32, gristmill_rewrites::repr::IndexId)> = Vec::new(); // old id -> new
    let mut exts: Vec<RIndex> = Vec::new();
    for &(_, old) in free_a.iter().chain(free_b.iter()) {
        let d = dom(
            &comp.definitions[first.def],
            &comp.definitions[first.def].rhs[first.term],
            old,
        );
        let nid = fresh(&mut next_id);
        ext_map.push((old, nid));
        exts.push(RIndex {
            id: nid,
            range: RangeId(d),
        });
    }
    let mut sums: Vec<RIndex> = Vec::new();
    let mut sum_map: BTreeMap<u32, gristmill_rewrites::repr::IndexId> = BTreeMap::new();
    for &old in &paired {
        let d = dom(
            &comp.definitions[first.def],
            &comp.definitions[first.def].rhs[first.term],
            old,
        );
        let nid = fresh(&mut next_id);
        sum_map.insert(old, nid);
        sums.push(RIndex {
            id: nid,
            range: RangeId(d),
        });
    }
    let remap = |old: u32,
                 ext_map: &[(u32, gristmill_rewrites::repr::IndexId)],
                 sum_map: &BTreeMap<u32, gristmill_rewrites::repr::IndexId>|
     -> gristmill_rewrites::repr::IndexId {
        if let Some((_, n)) = ext_map.iter().find(|(o, _)| *o == old) {
            *n
        } else {
            sum_map[&old]
        }
    };
    let intermediate_term = RTerm {
        sums,
        coeff: num_rational::BigRational::from_integer(1.into()),
        factors: vec![
            RTensorRef {
                tensor: TensorId(proposal.ta),
                indices: fa
                    .indices
                    .iter()
                    .map(|i| remap(i.0, &ext_map, &sum_map))
                    .collect(),
            },
            RTensorRef {
                tensor: TensorId(proposal.tb),
                indices: fb
                    .indices
                    .iter()
                    .map(|i| remap(i.0, &ext_map, &sum_map))
                    .collect(),
            },
        ],
    };
    if dbg { eprintln!("apply: exts/sums built"); }
    let intermediate_def = RTensorDef {
        base: TensorId(new_tid),
        exts: exts.clone(),
        rhs: vec![intermediate_term],
    };
    out.tensors.insert(
        TensorId(new_tid),
        TensorInfo {
            rank: exts.len(),
            symmetry: Vec::new(),
        },
    );
    // Rewrite member terms: replace the pair with the intermediate reference
    // (axes in canonical ext order).
    let mut touched_defs: std::collections::BTreeSet<usize> = Default::default();
    if dbg { eprintln!("apply: intermediate def built"); }
    for (oi, occ) in proposal.occs.iter().enumerate() {
        if dbg { eprintln!("apply: occ {oi}"); }
        let def = &mut out.definitions[occ.def];
        let term = &mut def.rhs[occ.term];
        if occ.fa >= term.factors.len() || occ.fb >= term.factors.len() {
            panic!(
                "occ {oi}: def{} term{} factors {} but pair ({},{})",
                occ.def, occ.term, term.factors.len(), occ.fa, occ.fb
            );
        }
        let fa = term.factors[occ.fa].clone();
        let fb = term.factors[occ.fb].clone();
        let paired_here: std::collections::BTreeSet<u32> = fa
            .indices
            .iter()
            .filter(|i| fb.indices.contains(i))
            .map(|i| i.0)
            .collect();
            // Map canonical ext axes to THIS term's index ids using the same
        // (domain, position) sort used for the canonical order.
        // Look domains up in the pristine input (ext/sum lists unchanged by
        // rewrites), avoiding borrow conflicts with the mutable target.
        let dom2 = |id: u32| -> u32 {
            dom(&comp.definitions[occ.def], &comp.definitions[occ.def].rhs[occ.term], id)
        };
        let pos_free_sorted = |f: &RTensorRef, other: &RTensorRef| -> Vec<usize> {
            let mut v: Vec<usize> = f
                .indices
                .iter()
                .enumerate()
                .filter(|(_, i)| !other.indices.contains(i))
                .map(|(p, _)| p)
                .collect();
            v.sort_by_key(|&p| (dom2(f.indices[p].0), p));
            v
        };
        let positions_a = pos_free_sorted(&fa, &fb);
        let positions_b = pos_free_sorted(&fb, &fa);
        let mut ref_indices = Vec::new();
        for &p in &positions_a {
            ref_indices.push(fa.indices[p]);
        }
        for &p in &positions_b {
            ref_indices.push(fb.indices[p]);
        }
        let new_factors: Vec<RTensorRef> = term
            .factors
            .iter()
            .enumerate()
            .filter(|(k, _)| *k != occ.fa && *k != occ.fb)
            .map(|(_, f)| f.clone())
            .chain(std::iter::once(RTensorRef {
                tensor: TensorId(new_tid),
                indices: ref_indices,
            }))
            .collect();
        term.factors = new_factors;
        term.sums
            .retain(|s| !paired_here.contains(&s.id.0));
        touched_defs.insert(occ.def);
    }
    // Insert the intermediate definition before the earliest touched def.
    let pos = touched_defs.iter().next().copied().unwrap_or(0);
    out.definitions.insert(pos, intermediate_def);
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn finds_groups_on_ccd() {
        let text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../benches/chemistry/ccd.json"
        ))
        .unwrap();
        let mut problem = truthmill::Problem::from_json(&text).unwrap();
        problem.sizes.insert(truthmill::DomainId(0), 4);
        problem.sizes.insert(truthmill::DomainId(1), 8);
        let (comp, _) = crate::bridge::problem_to_computation(&problem, 1000);
        let n0 = super::propose(&comp).len();
        let mut comp2 = comp.clone();
        crate::bridge::split_large_definitions(&mut comp2, 12);
        for (d, def) in comp2.definitions.iter().enumerate() {
            eprintln!("after split def{d}: {} terms", def.rhs.len());
        }
        let n1 = super::propose(&comp2).len();
        eprintln!("proposals before split: {n0}, after split: {n1}");
        let props = super::propose(&comp2);
        let (_, outputs) = crate::bridge::problem_to_computation(&problem, 1000);
        let s0 = gristmill_rewrites::state::State::new(comp2.clone(), outputs.clone()).unwrap();
        for (i, p) in props.iter().enumerate() {
            let c2 = super::apply_proposal(&comp2, p);
            assert!(c2.definitions.len() == comp2.definitions.len() + 1);
            let s1 = gristmill_rewrites::state::State::new(c2.clone(), outputs.clone()).unwrap();
            let eq = gristmill_rewrites::verify::equivalent_states(&s0, &s1).unwrap();
            eprintln!("prop{i} group{}: equivalent = {eq}", p.group_size);
            if !eq {
                eprintln!("ORIGINAL def terms of first occ:");
                let c2d = super::apply_proposal(&comp2, p);
                eprintln!("new defs: {}", c2d.definitions.len());
                for (di, d) in c2d.definitions.iter().enumerate() {
                    eprintln!(" def{di} base={} exts={:?} nterms={}", d.base.0, d.exts, d.rhs.len());
                    for (ti, t) in d.rhs.iter().enumerate() {
                        eprintln!("   t{ti}: factors {:?} sums {:?}", 
                            t.factors.iter().map(|f| (f.tensor.0, f.indices.iter().map(|x| x.0).collect::<Vec<_>>())).collect::<Vec<_>>(),
                            t.sums.iter().map(|s| s.id.0).collect::<Vec<_>>());
                    }
                }
            }
        }
    }
}
