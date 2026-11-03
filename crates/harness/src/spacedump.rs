//! I-007 space-vs-search discrimination: enumerate the kernel action space
//! beyond the greedy path, model-price ALL of it (no FLOP top-8 prefilter),
//! and compare: initial price vs baseline price vs greedy terminal vs the
//! best price found by budgeted best-first search.
//!
//! Verdict rule (pre-registered in IDEAS.md I-007):
//!   min(space) >= baseline price  => (a) the rewrite space has nothing
//!   min(space) < baseline but greedy terminal > min => (b) search problem
//!   prices invert vs measured anchors => (c) model problem (checked
//!   separately against exam/anchor numbers).

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use anyhow::Result;
use gristmill_rewrites::action::{self, ActionQuery, ActionSpace};
use gristmill_rewrites::repr::Computation;
use gristmill_rewrites::state::State;
use gristmill_rewrites::transition;
use truthmill::Problem;

use crate::{bridge, comp_hash, costmodel, dense, layout, layout3};

/// Enumerate every kernel action at `state` with its resulting state and
/// model price. Same action classes as the selector (biclique full +
/// leave-one-out variants, permutation, parenthesize cuts) but WITHOUT the
/// selector's symbolic-FLOP top-8 prefilter (I-007 prices the whole space).
fn enumerate_all(
    state: &State,
    score: &mut dyn FnMut(&Computation) -> f64,
    score_cache: &mut HashMap<u64, f64>,
) -> Vec<(f64, action::Action, String, State)> {
    let mut out = Vec::new();
    let mut cached_score = |comp: &Computation, cache: &mut HashMap<u64, f64>| -> f64 {
        let h = comp_hash(comp);
        if let Some(&c) = cache.get(&h) {
            return c;
        }
        let c = score(comp);
        cache.insert(h, c);
        c
    };
    let ndefs = state.computation().definitions.len();
    let biclique_max_terms: usize = std::env::var("H_BICLIQUE_MAX_TERMS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    for d in 0..ndefs {
        let nterms_d = state.computation().definitions[d].rhs.len();
        if nterms_d > biclique_max_terms {
            continue;
        }
        if let Ok(ActionSpace::Biclique(space)) =
            action::query(state, ActionQuery::BicliqueFactor(action::DefinitionPosition(d)))
        {
            let ncand = space.candidate_count();
            for ci in 0..ncand {
                if let Some((nl, nr)) = space.shape(ci) {
                    let mut masks: Vec<(Vec<bool>, Vec<bool>)> =
                        vec![(vec![true; nl], vec![true; nr])];
                    for i in 0..nl {
                        let mut l = vec![true; nl];
                        l[i] = false;
                        masks.push((l, vec![true; nr]));
                    }
                    for j in 0..nr {
                        let mut r = vec![true; nr];
                        r[j] = false;
                        masks.push((vec![true; nl], r));
                    }
                    for (left, right) in masks {
                        if let Ok(act) = space.select(ci, &left, &right) {
                            if let Ok(s2) = transition::apply(state, act.clone()) {
                                let c = cached_score(s2.computation(), score_cache);
                                out.push((c, act, format!("biclique def{d} cand{ci}"), s2));
                            }
                        }
                    }
                }
            }
        }
        if let Ok(ActionSpace::Permutation(space)) =
            action::query(state, ActionQuery::PermutationFactor(action::DefinitionPosition(d)))
        {
            let snaps = space.snapshot();
            for ci in 0..snaps.len() {
                let (_, roots, uses) = &snaps[ci];
                let rmask = vec![true; roots.len()];
                let umask = vec![true; uses.len()];
                if let Ok(act) = space.select(ci, &rmask, &umask) {
                    if let Ok(s2) = transition::apply(state, act.clone()) {
                        let c = cached_score(s2.computation(), score_cache);
                        out.push((c, act, format!("permute def{d} cand{ci}"), s2));
                    }
                }
            }
        }
        let nterms = state.computation().definitions[d].rhs.len();
        for tpos in 0..nterms {
            let nf = state.computation().definitions[d].rhs[tpos].factors.len();
            if nf < 3 {
                continue;
            }
            if let Ok(ActionSpace::Parenthesize(space)) =
                action::query(state, ActionQuery::Parenthesize(action::TermPosition {
                    definition: action::DefinitionPosition(d),
                    term: tpos,
                }))
            {
                for cut in 1..nf {
                    let mask: Vec<bool> = (0..nf).map(|i| i < cut).collect();
                    if let Ok(act) = space.select(&mask) {
                        if let Ok(s2) = transition::apply(state, act.clone()) {
                            let c = cached_score(s2.computation(), score_cache);
                            out.push((c, act, format!("paren def{d} term{tpos} cut{cut}"), s2));
                        }
                    }
                }
            }
        }
    }
    out
}

/// The selector's greedy trajectory under the same scoring (no prefilter),
/// for a same-instrument greedy terminal comparison.
fn greedy_terminal(
    mut state: State,
    score: &mut dyn FnMut(&Computation) -> f64,
    score_cache: &mut HashMap<u64, f64>,
    accept: f64,
    max_steps: usize,
) -> f64 {
    let mut cur = score(state.computation());
    for _ in 0..max_steps {
        let cands = enumerate_all(&state, score, score_cache);
        let best_idx = cands
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.0.partial_cmp(&b.0).unwrap())
            .map(|(i, _)| i);
        match best_idx {
            Some(idx) if cands[idx].0 < cur * (1.0 - accept) => {
                cur = cands[idx].0;
                state = cands.into_iter().nth(idx).unwrap().3;
            }
            _ => break,
        }
    }
    cur
}

pub fn run(
    problem: &Problem,
    name: &str,
    baseline_program: &truthmill::Program,
    threads: usize,
    max_expansions: usize,
    out: &Path,
) -> Result<()> {
    let rates = costmodel::rates_for_threads(threads);
    let input_ids: Vec<u32> = dense::collect_patterns(problem).keys().map(|t| t.0).collect();
    let (mut computation, outputs) = bridge::problem_to_computation(problem, 1000);
    // Match the selector's default reachable space (split_above = 20).
    bridge::split_large_definitions(&mut computation, 20);
    let output_ids: Vec<u32> = outputs.iter().map(|t| t.0).collect();

    let mut score = |comp: &Computation| -> f64 {
        let prog = bridge::computation_to_program(comp, &input_ids, &output_ids);
        let (bl, vl) = layout3::plan_layouts_model(problem, &prog, &rates, 1);
        let prog = layout::apply_layouts_to_program_v2(&prog, &vl);
        costmodel::program_costs(&prog, &problem.sizes, &bl, &rates)
            .iter()
            .sum()
    };
    // Baseline price through the same instrument.
    let baseline_price = {
        let (bl, vl) = layout3::plan_layouts_model(problem, baseline_program, &rates, 1);
        let prog = layout::apply_layouts_to_program_v2(baseline_program, &vl);
        costmodel::program_costs(&prog, &problem.sizes, &bl, &rates)
            .iter()
            .sum::<f64>()
    };

    let mut cache: HashMap<u64, f64> = HashMap::new();
    let comp0 = computation.clone();
    let outputs0 = outputs.clone();
    let state0 = State::new(computation, outputs).map_err(|e| anyhow::anyhow!("state: {e:?}"))?;
    let initial_price = score(state0.computation());

    // Greedy terminal (same instrument, no prefilter).
    let greedy_price = greedy_terminal(
        State::new(comp0, outputs0).map_err(|e| anyhow::anyhow!("state: {e:?}"))?,
        &mut score,
        &mut cache,
        0.005,
        40,
    );

    // Best-first over the action space with dedup and budget.
    let mut frontier: Vec<(f64, u64, State, usize)> = Vec::new(); // (score, hash, state, depth)
    let h0 = comp_hash(state0.computation());
    cache.insert(h0, initial_price);
    frontier.push((initial_price, h0, state0, 0));
    let mut seen: HashMap<u64, f64> = HashMap::new();
    seen.insert(h0, initial_price);
    let mut global_min = initial_price;
    let mut global_min_depth = 0usize;
    let mut expansion_log = Vec::new();
    for exp in 0..max_expansions {
        // pop min
        let (pos, _) = frontier
            .iter()
            .enumerate()
            .min_by(|a, b| a.1 .0.partial_cmp(&b.1 .0).unwrap())
            .unwrap();
        let (cur_score, cur_hash, cur_state, depth) = frontier.swap_remove(pos);
        let cands = enumerate_all(&cur_state, &mut score, &mut cache);
        let mut n_new = 0;
        for (c, _act, _desc, s2) in cands {
            let h = comp_hash(s2.computation());
            if seen.contains_key(&h) {
                continue;
            }
            seen.insert(h, c);
            n_new += 1;
            if c < global_min {
                global_min = c;
                global_min_depth = depth + 1;
            }
            frontier.push((c, h, s2, depth + 1));
        }
        expansion_log.push(serde_json::json!({
            "expansion": exp, "depth": depth, "score": cur_score, "hash": format!("{cur_hash:016x}"),
            "new_states": n_new, "frontier": frontier.len(), "global_min": global_min,
        }));
        eprintln!(
            "[space-dump] {name} exp{exp} depth {depth} score {cur_score:.4} new {n_new} frontier {} min {global_min:.4}",
            frontier.len()
        );
    }
    let verdict = if global_min >= baseline_price {
        "space"
    } else if greedy_price > global_min * 1.001 {
        "search"
    } else {
        "greedy-ok"
    };
    let rec = serde_json::json!({
        "kind": "space-dump",
        "problem": name,
        "threads": threads,
        "sizes": crate::sizes_map(problem),
        "initial_price_s": initial_price,
        "baseline_price_s": baseline_price,
        "greedy_terminal_s": greedy_price,
        "best_first_min_s": global_min,
        "best_first_min_depth": global_min_depth,
        "expansions": expansion_log.len(),
        "states_seen": seen.len(),
        "verdict_model_priced": verdict,
        "log": expansion_log,
    });
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(out)?;
    writeln!(f, "{}", serde_json::to_string(&rec)?)?;
    println!("{}", serde_json::to_string(&rec)?);
    Ok(())
}
