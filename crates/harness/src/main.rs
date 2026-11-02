//! Harness: correctness oracle and steady-state timing for truthmill problems
//! executed on tenferro.

mod bridge;
mod costmodel;
mod propose;
mod dense;
mod interp;
mod layout;
mod layout2;
mod layout3;
mod oracle;
mod transform;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use tenferro_cpu::CpuBackend;
use truthmill::{DomainId, Problem};

#[derive(Parser)]
#[command(name = "harness")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Validate the tenferro interpreter against the independent scalar oracle.
    Check {
        problem: PathBuf,
        /// Size overrides, e.g. "0=2,1=3".
        #[arg(long)]
        sizes: Option<String>,
        #[arg(long, default_value = "0")]
        seed: u64,
        /// Program transform to apply before evaluation: none|greedy
        #[arg(long, default_value = "none")]
        transform: String,
        /// External candidate Program JSON (e.g. gristmill-optimized); skips transform
        #[arg(long)]
        candidate: Option<PathBuf>,
        /// Layout planning: none|greedy-layout
        #[arg(long, default_value = "none")]
        layout: String,
        /// Mismatch penalty for layout cost model
        #[arg(long, default_value = "2.7")]
        penalty: f64,
    },
    /// Steady-state timing of the reference program with prepared plans.
    Time {
        problem: PathBuf,
        #[arg(long)]
        sizes: Option<String>,
        #[arg(long, default_value = "1")]
        threads: usize,
        #[arg(long, default_value = "10")]
        iters: usize,
        #[arg(long, default_value = "3")]
        warmup: usize,
        #[arg(long, default_value = "0")]
        seed: u64,
        /// Program transform to apply before evaluation: none|greedy
        #[arg(long, default_value = "none")]
        transform: String,
        /// Layout planning: none|greedy-layout
        #[arg(long, default_value = "none")]
        layout: String,
        /// Use the cold interpreter (replan every call) instead of prepared plans.
        #[arg(long)]
        cold: bool,
        /// Mismatch penalty for layout cost model (big ops)
        #[arg(long, default_value = "2.7")]
        penalty: f64,
        /// Mismatch penalty for small ops (below --p-thresh FLOPs)
        #[arg(long)]
        p_small: Option<f64>,
        /// FLOPs threshold separating small/big penalty regimes
        #[arg(long, default_value = "1e8")]
        p_thresh: f64,
        /// Refuse to run if the static memory estimate exceeds this (MB).
        #[arg(long, default_value = "6000")]
        max_mem_mb: usize,
        /// factmem: split n-ary einsums whose union product exceeds this (MB).
        #[arg(long, default_value = "256")]
        factmem_threshold_mb: u64,
        /// External candidate Program JSON (e.g. gristmill-optimized); skips transform
        #[arg(long)]
        candidate: Option<PathBuf>,
    },
    /// HL search: forward selection over shared-intermediate candidates with
    /// MEASURED acceptance (small-size wall time), exact verify per step.
    Search {
        problem: PathBuf,
        #[arg(long)]
        sizes: Option<String>,
        #[arg(long, default_value = "1")]
        threads: usize,
        #[arg(long, default_value = "0")]
        seed: u64,
        /// Max accepted intermediates.
        #[arg(long, default_value = "30")]
        budget: usize,
        /// Candidates considered per round (by FLOP-saving order).
        #[arg(long, default_value = "12")]
        topk: usize,
        /// Output path for the best program found (truthmill Program JSON).
        #[arg(long)]
        out: PathBuf,
        /// Evaluate candidates with layout planning applied (slower, realistic).
        #[arg(long)]
        search_layout: bool,
    },
    /// Print symbolic FLOP terms for reference and greedy-factored programs.
    Flops {
        problem: PathBuf,
        /// Evaluate polynomials numerically at these sizes, e.g. "0=100,1=1000".
        #[arg(long)]
        sizes: Option<String>,
    },
    /// Validate the analytic cost model: per-op predicted vs measured times.
    CostCheck {
        problem: PathBuf,
        #[arg(long)]
        sizes: Option<String>,
        #[arg(long, default_value = "1")]
        threads: usize,
        #[arg(long, default_value = "3")]
        iters: usize,
        #[arg(long, default_value = "0")]
        seed: u64,
        #[arg(long, default_value = "none")]
        transform: String,
        /// External candidate Program JSON; skips transform
        #[arg(long)]
        candidate: Option<PathBuf>,
        #[arg(long, default_value = "none")]
        layout: String,
    },
    /// Dump the expanded rewrites-Computation JSON for a problem.
    DumpComp {
        problem: PathBuf,
        /// Only split definitions larger than this many terms (0 = never).
        #[arg(long, default_value = "0")]
        split_above: usize,
    },
    /// Probe: time kernel query() per definition, no selector logic.
    QueryProbe {
        problem: PathBuf,
    },
    /// HL selector: greedy rewrite search over gristmill-rewrites kernel,
    /// scored by the analytic wall-time model (layout-in-the-loop).
    Select {
        problem: PathBuf,
        #[arg(long)]
        sizes: Option<String>,
        #[arg(long, default_value = "1")]
        threads: usize,
        #[arg(long, default_value = "40")]
        max_steps: usize,
        #[arg(long)]
        out: Option<PathBuf>,
        /// Minimum relative improvement to accept a step.
        #[arg(long, default_value = "0.005")]
        accept: f64,
        /// Selector scoring: model (wall-time, ours) | flops (FLOP-greedy,
        /// replicates gristmill's selection policy on the same kernel spaces)
        #[arg(long, default_value = "model")]
        score: String,
        /// Engine: kernel | structural | hybrid (structural for defs >20,
        /// then kernel for the remainder; verify structural steps)
        #[arg(long, default_value = "kernel")]
        engine: String,
        /// Experimental bounded parenthesize-then-action lookahead (I-005).
        #[arg(long)]
        lookahead: bool,
        /// Split definitions above this many terms (0 disables splitting).
        #[arg(long, default_value = "20")]
        split_above: usize,
        /// Same-equation gristmill baseline used as the Axis-1 cap/fallback.
        /// Defaults to ../candidates/<problem>_gristmill.json.
        #[arg(long)]
        baseline: Option<PathBuf>,
    },
    /// Per-op timing profile of one program execution (prepared plans).
    Profile {
        problem: PathBuf,
        #[arg(long)]
        sizes: Option<String>,
        #[arg(long, default_value = "1")]
        threads: usize,
        #[arg(long, default_value = "5")]
        iters: usize,
        #[arg(long, default_value = "0")]
        seed: u64,
        #[arg(long, default_value = "none")]
        transform: String,
        /// External candidate Program JSON; skips transform
        #[arg(long)]
        candidate: Option<PathBuf>,
        /// Layout planning: none|greedy-layout
        #[arg(long, default_value = "none")]
        layout: String,
    },
    /// Decomposed run report: whole-program timing + per-op panel with model
    /// predictions, validity footer, and optional pinned-anchor comparison.
    Report {
        problem: PathBuf,
        #[arg(long)]
        sizes: Option<String>,
        #[arg(long, default_value = "1")]
        threads: usize,
        #[arg(long, default_value = "5")]
        iters: usize,
        #[arg(long, default_value = "1")]
        warmup: usize,
        #[arg(long, default_value = "0")]
        seed: u64,
        #[arg(long, default_value = "none")]
        transform: String,
        /// External candidate Program JSON; skips transform
        #[arg(long)]
        candidate: Option<PathBuf>,
        #[arg(long, default_value = "model")]
        layout: String,
        #[arg(long, default_value = "2.7")]
        penalty: f64,
        /// Store this run as a pinned anchor profile at this path.
        #[arg(long)]
        store: Option<PathBuf>,
        /// Compare against a previously stored anchor profile.
        #[arg(long)]
        anchor: Option<PathBuf>,
        /// Max ops in the top list; rest is aggregated.
        #[arg(long, default_value = "15")]
        top: usize,
        /// Append the run record as a JSON line to this file.
        #[arg(long)]
        log: Option<PathBuf>,
    },
}

fn best_kernel_action(
    state: &gristmill_rewrites::state::State,
    score: &dyn Fn(&gristmill_rewrites::repr::Computation) -> f64,
    log_sizes: &BTreeMap<gristmill_rewrites::repr::RangeId, f64>,
    dbg: bool,
) -> Option<(f64, gristmill_rewrites::action::Action, String)> {
    use gristmill_rewrites::action::{self, ActionQuery, ActionSpace};
    use gristmill_rewrites::transition;

    let mut best: Option<(f64, action::Action, String)> = None;
    let ndefs = state.computation().definitions.len();
    let biclique_max_terms: usize = std::env::var("H_BICLIQUE_MAX_TERMS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    for d in 0..ndefs {
        let dstart = std::time::Instant::now();
        let nterms_d = state.computation().definitions[d].rhs.len();
        if nterms_d > biclique_max_terms {
            // Kernel enumerates ALL maximal bicliques (exponential, no cutoff):
            // skip large definitions until the kernel grows an enumeration budget.
            continue;
        }
        if dbg {
            eprintln!("[dbg] def{d}: {nterms_d} terms, query start");
        }
        // Biclique, with tiered prefilter: rank candidates by the kernel's
        // symbolic log_flops, model-price only top-k.
        if let Ok(ActionSpace::Biclique(space)) = action::query(
            state,
            ActionQuery::BicliqueFactor(action::DefinitionPosition(d)),
        ) {
            let base_flops = gristmill_rewrites::cost::log_flops(
                state.computation(),
                log_sizes,
            )
            .unwrap_or(f64::INFINITY);
            let mut priced: Vec<(f64, usize, action::Action)> = Vec::new();
            let ncand = space.candidate_count();
            if dbg {
                eprintln!("[dbg] def{d}: {ncand} biclique candidates");
            }
            for ci in 0..ncand {
                if let Some((nl, nr)) = space.shape(ci) {
                    // Full fire + leave-one-out variants.
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
                                let f2 = gristmill_rewrites::cost::log_flops(
                                    s2.computation(),
                                    log_sizes,
                                )
                                .unwrap_or(f64::INFINITY);
                                if dbg {
                                    eprintln!("[dbg]   cand{ci} dflops={:.3}", f2 - base_flops);
                                }
                                priced.push((f2 - base_flops, ci, act));
                            }
                        }
                    }
                }
            }
            priced.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            for (_, ci, act) in priced.into_iter().take(8) {
                if let Ok(s2) = transition::apply(state, act.clone()) {
                    let c = score(s2.computation());
                    let better = match &best {
                        None => true,
                        Some((b, _, _)) => c < *b,
                    };
                    if better {
                        best = Some((c, act, format!("biclique def{d} cand{ci}")));
                    }
                }
            }
        }
        if dbg {
            eprintln!("[dbg] def{d} biclique phase: {:?}", dstart.elapsed());
        }
        // Permutation.
        if let Ok(ActionSpace::Permutation(space)) = action::query(
            state,
            ActionQuery::PermutationFactor(action::DefinitionPosition(d)),
        ) {
            let snaps = space.snapshot();
            for ci in 0..snaps.len() {
                let (_, roots, uses) = &snaps[ci];
                let rmask = vec![true; roots.len()];
                let umask = vec![true; uses.len()];
                if let Ok(act) = space.select(ci, &rmask, &umask) {
                    if let Ok(s2) = transition::apply(state, act.clone()) {
                        let c = score(s2.computation());
                        let better = match &best {
                            None => true,
                            Some((b, _, _)) => c < *b,
                        };
                        if better {
                            best = Some((c, act, format!("permute def{d} cand{ci}")));
                        }
                    }
                }
            }
        }
        if dbg {
            eprintln!("[dbg] def{d} permutation phase: {:?}", dstart.elapsed());
        }
        // Parenthesize splits of multi-factor terms.
        let nterms = state.computation().definitions[d].rhs.len();
        for tpos in 0..nterms {
            let nf = state.computation().definitions[d].rhs[tpos].factors.len();
            if nf < 3 {
                continue;
            }
            if let Ok(ActionSpace::Parenthesize(space)) =
                action::query(&state, ActionQuery::Parenthesize(action::TermPosition {
                    definition: action::DefinitionPosition(d),
                    term: tpos,
                }))
            {
                for cut in 1..nf {
                    let mask: Vec<bool> = (0..nf).map(|i| i < cut).collect();
                    if let Ok(act) = space.select(&mask) {
                        if let Ok(s2) = transition::apply(state, act.clone()) {
                            let c = score(s2.computation());
                            if dbg {
                                eprintln!(
                                    "[dbg]   paren def{d} term{tpos} cut{cut}: {c:.6} s"
                                );
                            }
                            let better = match &best {
                                None => true,
                                Some((b, _, _)) => c < *b,
                            };
                            if better {
                                best = Some((
                                    c,
                                    act,
                                    format!("paren def{d} term{tpos} cut{cut}"),
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
    best
}

fn parse_sizes(problem: &mut Problem, overrides: &Option<String>) -> Result<()> {
    if let Some(overrides) = overrides {
        for part in overrides.split(',') {
            let (k, v) = part
                .split_once('=')
                .with_context(|| format!("bad size override: {part}"))?;
            problem
                .sizes
                .insert(DomainId(k.parse()?), v.parse::<u64>()?);
        }
    }
    Ok(())
}

fn stem(path: &PathBuf) -> String {
    path.file_stem().unwrap().to_string_lossy().to_string()
}

fn load_problem(path: &PathBuf, sizes: &Option<String>) -> Result<Problem> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {path:?}"))?;
    let mut problem = Problem::from_json(&text).map_err(anyhow::Error::msg)?;
    parse_sizes(&mut problem, sizes)?;
    truthmill::validate(&problem).map_err(|e| anyhow::anyhow!("validation: {e:?}"))?;
    Ok(problem)
}

fn main() -> Result<()> {
    // Hard address-space backstop (default 10 GB, override with H_RLIMIT_GB):
    // a runaway allocation aborts this process instead of the machine.
    let limit_gb: u64 = std::env::var("H_RLIMIT_GB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    unsafe {
        let bytes = limit_gb * (1u64 << 30);
        let rl = libc::rlimit {
            rlim_cur: bytes,
            rlim_max: bytes,
        };
        libc::setrlimit(libc::RLIMIT_AS, &rl);
    }
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Check {
            problem,
            sizes,
            seed,
            transform,
            candidate,
            layout,
            penalty,
        } => {
            let name = stem(&problem);
            let problem = load_problem(&problem, &sizes)?;
            let program = load_program(&problem, &transform, 256, candidate)?;
            let orig = dense::generate_inputs_for(&problem, &[&program], seed);
            let (program, blocks) = apply_layout_with(&problem, program, &layout, orig,
                &layout2::PolicyParams::default(), 1)?;
            let expected = oracle::eval_program(&problem, &problem.reference, &blocks.orig)?;
            let tensor_blocks = interp::to_tensor_blocks_permuted(&blocks.permuted)?;
            let mut backend = CpuBackend::with_threads(1).map_err(anyhow::Error::msg)?;
            let actual = interp::run_cold(&program, &tensor_blocks, &mut backend)?;
            let mut worst = 0.0f64;
            let mut report = Vec::new();
            for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
                anyhow::ensure!(
                    e.dims == a.shape(),
                    "output {i}: shape mismatch {:?} vs {:?}",
                    e.dims,
                    a.shape()
                );
                let data = a.as_slice::<f64>().map_err(anyhow::Error::msg)?;
                let mut max_abs = 0.0f64;
                let mut scale = 0.0f64;
                for (x, y) in e.data.iter().zip(data) {
                    max_abs = max_abs.max((x - y).abs());
                    scale = scale.max(x.abs()).max(y.abs());
                }
                let rel = if scale > 0.0 { max_abs / scale } else { 0.0 };
                worst = worst.max(rel);
                report.push(serde_json::json!({
                    "output": i,
                    "shape": e.dims,
                    "max_abs_diff": max_abs,
                    "max_rel_diff": rel,
                }));
            }
            let record = serde_json::json!({
                "kind": "check",
                "transform": transform,
                "layout": layout,
                "problem": name,
                "sizes": sizes_map(&problem),
                "seed": seed,
                "outputs": report,
                "worst_rel_diff": worst,
                "pass": worst < 1e-10,
            });
            println!("{}", serde_json::to_string(&record)?);
            if worst >= 1e-10 {
                anyhow::bail!("oracle check failed: worst rel diff {worst:e}");
            }
        }
        Cmd::Time {
            problem,
            sizes,
            threads,
            iters,
            warmup,
            seed,
            transform,
            layout,
            cold,
            penalty,
            p_small,
            p_thresh,
            max_mem_mb,
            factmem_threshold_mb,
            candidate,
        } => {
            let name = stem(&problem);
            let problem = load_problem(&problem, &sizes)?;
            let input_mb = dense::estimate_input_bytes(&problem) / (1 << 20);
            if input_mb > max_mem_mb / 2 {
                anyhow::bail!(
                    "memory guard: inputs alone need {input_mb} MB (limit {} MB); shrink sizes",
                    max_mem_mb
                );
            }
            let cand_stem = candidate.as_ref().map(|p| stem(p)).unwrap_or_default();
            let program = load_program(&problem, &transform, factmem_threshold_mb, candidate)?;
            let orig = dense::generate_inputs_for(&problem, &[&program], seed);
            let params = layout2::PolicyParams {
                p_big: penalty,
                p_small: p_small.unwrap_or(penalty),
                thresh: p_thresh,
            };
            let (program, blocks) = apply_layout_with(&problem, program, &layout, orig, &params, threads)?;
            let tensor_blocks = interp::to_tensor_blocks_permuted(&blocks.permuted)?;
            let est = interp::estimate_memory(&program, &tensor_blocks, &problem.sizes, 1 << 30);
            if !est.risky_ops.is_empty() {
                eprintln!(
                    "WARNING: {} n-ary einsums with >1GB union intermediates: {:?}                      (backend-internal temporaries may exceed estimates)",
                    est.risky_ops.len(),
                    est.risky_ops
                );
            }
            let total_mb = est.peak_live_bytes / (1 << 20);
            if total_mb > max_mem_mb {
                eprintln!(
                    "REFUSED: estimated peak {} MB (inputs {} MB) vs limit {} MB; risky ops: {:?}",
                    total_mb,
                    est.input_bytes / (1 << 20),
                    max_mem_mb,
                    est.risky_ops
                );
                anyhow::bail!("memory guard tripped");
            }
            drop(blocks); // free original + dense copies before timing
            let mut backend =
                CpuBackend::with_threads(threads).map_err(anyhow::Error::msg)?;
            let log_flops = truthmill::log_flops(&problem, &program)
                .map_err(|e| anyhow::anyhow!("cost: {e:?}"))?;
            let plans = if cold {
                None
            } else {
                Some(interp::prepare(&program, &tensor_blocks)?)
            };
            let mut times = Vec::with_capacity(iters);
            let mut sum = 0.0;
            for i in 0..(warmup + iters) {
                let start = Instant::now();
                let outputs = match &plans {
                    Some(plans) => {
                        interp::run_prepared(&program, plans, &tensor_blocks, &mut backend)?
                    }
                    None => interp::run_cold(&program, &tensor_blocks, &mut backend)?,
                };
                let elapsed = start.elapsed().as_secs_f64() * 1e3;
                sum += interp::checksum(&outputs);
                if i >= warmup {
                    times.push(elapsed);
                }
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = times[times.len() / 2];
            let record = serde_json::json!({
                "kind": "time",
                "transform": transform,
                "candidate": cand_stem,
                "layout": layout,
                "problem": name,
                "sizes": sizes_map(&problem),
                "threads": threads,
                "mode": if cold { "cold" } else { "prepared" },
                "iters": iters,
                "warmup": warmup,
                "median_ms": median,
                "min_ms": times[0],
                "max_ms": times[times.len() - 1],
                "log10_flops": log_flops / std::f64::consts::LN_10,
                "p_big": penalty,
                "p_small": p_small.unwrap_or(penalty),
                "p_thresh": p_thresh,
                "checksum": sum,
            });
            println!("{}", serde_json::to_string(&record)?);
        }
        Cmd::Search {
            problem,
            sizes,
            threads,
            seed,
            budget,
            topk,
            out,
            search_layout,
        } => {
            let name = stem(&problem);
            let problem = load_problem(&problem, &sizes)?;
            let orig = dense::generate_inputs(&problem, seed);
            let tensor_blocks = interp::to_tensor_blocks(&orig)?;
            let mut backend = CpuBackend::with_threads(threads).map_err(anyhow::Error::msg)?;

            let time_program = |prog: &truthmill::Program,
                                blocks_in: &interp::TensorBlocks,
                                backend: &mut CpuBackend|
             -> Result<f64> {
                // Optionally plan layouts on the candidate and permute blocks
                // accordingly, so acceptance reflects layout-adjusted cost.
                let (program_cow, blocks_owned);
                let program: &truthmill::Program;
                let blocks: &interp::TensorBlocks;
                if search_layout {
                    let (bl, vals) = layout2::plan_layouts_v2(&problem, prog,
                        &layout2::PolicyParams::default());
                    program_cow = layout::apply_layouts_to_program_v2(prog, &vals);
                    let pb = layout::apply_layouts_to_blocks(&orig, &bl);
                    blocks_owned = interp::to_tensor_blocks_permuted(&pb)?;
                    program = &program_cow;
                    blocks = &blocks_owned;
                } else {
                    program = prog;
                    blocks = blocks_in;
                }
                let est = interp::estimate_memory(program, blocks, &problem.sizes, 1 << 30);
                if est.peak_live_bytes / (1 << 20) > 6000 {
                    return Ok(f64::INFINITY);
                }
                let plans = interp::prepare(program, blocks)?;
                let mut times = Vec::new();
                for i in 0..4 {
                    let start = Instant::now();
                    let outputs = interp::run_prepared(program, &plans, blocks, backend)?;
                    std::hint::black_box(interp::checksum(&outputs));
                    if i >= 1 {
                        times.push(start.elapsed().as_secs_f64() * 1e3);
                    }
                }
                times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                Ok(times[times.len() / 2])
            };

            let mut current = problem.reference.clone();
            let mut best_ms = time_program(&current, &tensor_blocks, &mut backend)?;
            println!("{}", serde_json::json!({"kind":"search","problem":name,"round":0,"best_ms":best_ms}));
            let mut accepted = 0;
            let mut round = 0;
            while accepted < budget {
                round += 1;
                let groups = transform::pair_occurrences(&current, &problem.sizes);
                let mut cand: Vec<_> = groups
                    .into_values()
                    .filter(|o| o.len() >= 2)
                    .collect();
                cand.sort_by(|a, b| {
                    let sa = (a.len() - 1) as f64 * a[0].formation_flops;
                    let sb = (b.len() - 1) as f64 * b[0].formation_flops;
                    sb.partial_cmp(&sa).unwrap()
                });
                let mut improved = false;
                for occs in cand.into_iter().take(topk) {
                    // skip oversized intermediates
                    let first = &occs[0];
                    let op = match &current.values[first.op] {
                        truthmill::ValueDef::Einsum(op) => op,
                        _ => continue,
                    };
                    let elems: usize = first
                        .inter_order
                        .iter()
                        .map(|c| problem.sizes[&op.code.domains[&first.local_of_canon[c]]] as usize)
                        .product();
                    if elems * 8 > (256 << 20) {
                        continue;
                    }
                    let trial = transform::share_introduce(&current, &occs);
                    let ok = truthmill::verify(&problem, &trial)
                        .map_err(|e| anyhow::anyhow!("verify error: {e:?}"))?;
                    if !ok {
                        println!("{}", serde_json::json!({"kind":"search-reject-verify","round":round}));
                        continue;
                    }
                    let ms = time_program(&trial, &tensor_blocks, &mut backend)?;
                    if ms < best_ms * 0.995 {
                        println!("{}", serde_json::json!({"kind":"search","round":round,"accepted":accepted+1,"best_ms":ms,"prev_ms":best_ms,"group_size":occs.len()}));
                        best_ms = ms;
                        current = trial;
                        accepted += 1;
                        improved = true;
                        break; // recompute groups after structural change
                    }
                }
                if !improved {
                    println!("{}", serde_json::json!({"kind":"search","round":round,"note":"no improving candidate","best_ms":best_ms}));
                    break;
                }
            }
            // exact verify + save
            let ok = truthmill::verify(&problem, &current)
                .map_err(|e| anyhow::anyhow!("verify error: {e:?}"))?;
            anyhow::ensure!(ok);
            std::fs::write(&out, serde_json::to_string(&current)?)?;
            println!("{}", serde_json::json!({"kind":"search-done","problem":name,"accepted":accepted,"best_ms":best_ms,"out":format!("{}",out.display())}));
        }
        Cmd::Flops { problem, sizes } => {
            let name = stem(&problem);
            let mut problem = load_problem(&problem, &sizes)?;
            // evaluate polynomials at requested sizes but keep original for display
            let greedy = transform::greedy_factor(&problem.reference, &problem.sizes);
            for (tag, prog) in [("reference", &problem.reference), ("greedy", &greedy)] {
                let sf = truthmill::symbolic_flops(&problem, prog)
                    .map_err(|e| anyhow::anyhow!("cost: {e:?}"))?;
                let logf = truthmill::log_flops(&problem, prog)
                    .map_err(|e| anyhow::anyhow!("cost: {e:?}"))?;
                println!("{name} {tag}: ln_flops={:.2}", logf);
                for t in sf.terms() {
                    let mut mon = String::new();
                    for (d, e) in t.powers() {
                        mon.push_str(&format!("d{}^{} ", d.0, e));
                    }
                    println!("   coeff={} {}", t.coefficient(), mon);
                }
            }
        }
        Cmd::CostCheck {
            problem,
            sizes,
            threads,
            iters,
            seed,
            transform,
            candidate,
            layout,
        } => {
            let name = stem(&problem);
            let problem = load_problem(&problem, &sizes)?;
            if dense::estimate_input_bytes(&problem) / (1 << 20) > 3000 {
                anyhow::bail!("memory guard: inputs too large; shrink sizes");
            }
            let program = load_program(&problem, &transform, 256, candidate)?;
            let orig = dense::generate_inputs_for(&problem, &[&program], seed);
            let (program, blocks) = apply_layout_with(
                &problem,
                program,
                &layout,
                orig,
                &layout2::PolicyParams::default(),
                threads,
            )?;
            let tensor_blocks = interp::to_tensor_blocks_permuted(&blocks.permuted)?;
            let est = interp::estimate_memory(&program, &tensor_blocks, &problem.sizes, 1 << 30);
            if est.peak_live_bytes / (1 << 20) > 6000 {
                anyhow::bail!("memory guard tripped");
            }
            let rates = costmodel::rates_for_threads(threads);
            let predicted = costmodel::program_costs(&program, &problem.sizes, &blocks.layouts, &rates);
            let mut backend = CpuBackend::with_threads(threads).map_err(anyhow::Error::msg)?;
            let measured =
                interp::profile_ops(&program, &tensor_blocks, &mut backend, iters, &problem.sizes)?;
            let mut rows = Vec::new();
            for (p, m) in predicted.iter().zip(&measured) {
                rows.push((p * 1e3, m.median_ms)); // both ms
            }
            let rho = spearman(&rows);
            println!("problem {name} threads {threads}: spearman = {rho:.3}, n = {}", rows.len());
            let mut sorted = rows.clone();
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            println!("top-10 by measured: (measured_ms, predicted_ms)");
            for (p, m) in sorted.iter().take(10) {
                println!("  {m:9.2}  {p:9.2}");
            }
            let sum_p: f64 = rows.iter().map(|r| r.0).sum();
            let sum_m: f64 = rows.iter().map(|r| r.1).sum();
            println!("total measured {sum_m:.1} ms, predicted {sum_p:.1} ms");
        }
        Cmd::DumpComp { problem, split_above } => {
            let problem = load_problem(&problem, &None)?;
            let (mut computation, _outputs) = bridge::problem_to_computation(&problem, 1000);
            if split_above > 0 {
                bridge::split_large_definitions(&mut computation, split_above);
            }
            println!(
                "{}",
                serde_json::to_string(&computation).map_err(anyhow::Error::msg)?
            );
        }
        Cmd::QueryProbe { problem } => {
            let problem = load_problem(&problem, &None)?;
            let (computation, outputs) = bridge::problem_to_computation(&problem, 1000);
            eprintln!("bridge done: {} definitions", computation.definitions.len());
            let t0 = std::time::Instant::now();
            let state = gristmill_rewrites::state::State::new(computation, outputs)
                .map_err(|e| anyhow::anyhow!("state: {e:?}"))?;
            eprintln!("state::new (canonicalize+validate): {:.2?}", t0.elapsed());
            for d in 0..state.computation().definitions.len() {
                let nterms = state.computation().definitions[d].rhs.len();
                // permutation space probe
                let t0 = std::time::Instant::now();
                let rp = gristmill_rewrites::action::query(
                    &state,
                    gristmill_rewrites::action::ActionQuery::PermutationFactor(
                        gristmill_rewrites::action::DefinitionPosition(d),
                    ),
                );
                let el = t0.elapsed();
                match rp {
                    Ok(gristmill_rewrites::action::ActionSpace::Permutation(space)) => {
                        println!(
                            "def{d}: permutation query {el:.2?}, {} families",
                            space.snapshot().len()
                        );
                    }
                    Ok(_) => println!("def{d}: non-permutation space"),
                    Err(e) => println!("def{d}: permutation query error {e:?} after {el:.2?}"),
                }
let t0 = std::time::Instant::now();
                let r = gristmill_rewrites::action::query(
                    &state,
                    gristmill_rewrites::action::ActionQuery::BicliqueFactor(
                        gristmill_rewrites::action::DefinitionPosition(d),
                    ),
                );
                let el = t0.elapsed();
                match r {
                    Ok(gristmill_rewrites::action::ActionSpace::Biclique(space)) => {
                        println!(
                            "def{d}: {nterms} terms, query {:.2?}, {} candidates",
                            el,
                            space.candidate_count()
                        );
                    }
                    Ok(_) => println!("def{d}: non-biclique space"),
                    Err(e) => println!("def{d}: query error {e:?} after {el:.2?}"),
                }
                            }
        }
        Cmd::Select {
            problem,
            sizes,
            threads,
            max_steps,
            out,
            accept,
            score: score_mode,
            engine,
            lookahead,
            split_above,
            baseline,
        } => {
            let name = stem(&problem);
            let baseline_path = baseline.unwrap_or_else(|| {
                problem
                    .parent()
                    .unwrap()
                    .join("../candidates")
                    .join(format!("{name}_gristmill.json"))
            });
            let problem = load_problem(&problem, &sizes)?;
            let baseline_program = load_program(
                &problem,
                "none",
                256,
                Some(baseline_path.clone()),
            )?;
            let baseline_log_flops = truthmill::log_flops(&problem, &baseline_program)
                .map_err(|e| anyhow::anyhow!("baseline flops: {e:?}"))?;
            let flop_cap_log = baseline_log_flops + 1.05f64.ln();
            let input_ids: Vec<u32> = dense::collect_patterns(&problem)
                .keys()
                .map(|t| t.0)
                .collect();
            let (mut computation, outputs) = bridge::problem_to_computation(&problem, 1000);
            if split_above > 0 && engine != "hybrid" {
                bridge::split_large_definitions(&mut computation, split_above);
            }
            let output_ids: Vec<u32> = outputs.iter().map(|t| t.0).collect();
            let rates = costmodel::rates_for_threads(threads);

            let log_sizes: std::collections::BTreeMap<gristmill_rewrites::repr::RangeId, f64> =
                problem
                    .sizes
                    .iter()
                    .map(|(dom, s)| (gristmill_rewrites::repr::RangeId(dom.0), (*s as f64).ln()))
                    .collect();
            let score = |comp: &gristmill_rewrites::repr::Computation| -> f64 {
                if score_mode == "flops" {
                    // gristmill-policy replica: symbolic FLOPs only.
                    return gristmill_rewrites::cost::log_flops(comp, &log_sizes)
                        .unwrap_or(f64::INFINITY);
                }
                let prog =
                    bridge::computation_to_program(comp, &input_ids, &output_ids);
                let (bl, vl) = layout3::plan_layouts_model(&problem, &prog, &rates, 1);
                let prog = layout::apply_layouts_to_program_v2(&prog, &vl);
                costmodel::program_costs(&prog, &problem.sizes, &bl, &rates)
                    .iter()
                    .sum()
            };

            use gristmill_rewrites::action::{self, ActionQuery, ActionSpace};
            use gristmill_rewrites::state::State;
            use gristmill_rewrites::transition;

            if engine == "structural" {
                return run_structural(
                    &problem, &name, &input_ids, &output_ids, &outputs, &rates,
                    threads, max_steps, accept, out, &baseline_program, flop_cap_log,
                );
            }
            if engine == "hybrid" {
                computation = optimize_large_definitions(
                    &computation,
                    &outputs,
                    &score,
                    max_steps,
                    accept,
                )?;
            }
            let mut state = State::new(computation, outputs.clone())
                .map_err(|e| anyhow::anyhow!("state: {e:?}"))?;
            let mut cur = score(state.computation());
            eprintln!("[select] initial predicted {cur:.4} s");
            let mut trace = Vec::new();
            for step in 0..max_steps {
                let dbg = std::env::var("H_DEBUG").is_ok();
                let step_start = std::time::Instant::now();
                let mut best: Option<(f64, action::Action, String)> = None;
                let ndefs = state.computation().definitions.len();
                let biclique_max_terms: usize = std::env::var("H_BICLIQUE_MAX_TERMS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(20);
                for d in 0..ndefs {
                    let dstart = std::time::Instant::now();
                    let nterms_d = state.computation().definitions[d].rhs.len();
                    if nterms_d > biclique_max_terms {
                        // Kernel enumerates ALL maximal bicliques (exponential,
                        // no cutoff): skip large definitions until the kernel
                        // grows an enumeration budget.
                        continue;
                    }
                    if dbg {
                        eprintln!("[dbg] def{d}: {nterms_d} terms, query start");
                    }
                    // biclique, with tiered prefilter: rank candidates by the
                    // kernel's symbolic log_flops, model-price only top-k.
                    if let Ok(ActionSpace::Biclique(space)) =
                        action::query(&state, ActionQuery::BicliqueFactor(action::DefinitionPosition(d)))
                    {
                        let base_flops = gristmill_rewrites::cost::log_flops(
                            state.computation(),
                            &log_sizes,
                        )
                        .unwrap_or(f64::INFINITY);
                        let mut priced: Vec<(f64, usize, action::Action)> = Vec::new();
                        let ncand = space.candidate_count();
                        if std::env::var("H_DEBUG").is_ok() {
                            eprintln!("[dbg] def{d}: {ncand} biclique candidates");
                        }
                        for ci in 0..ncand {
                            if let Some((nl, nr)) = space.shape(ci) {
                              // full fire + leave-one-out variants
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
                                    if let Ok(s2) = transition::apply(&state, act.clone()) {
                                        let f2 = gristmill_rewrites::cost::log_flops(
                                            s2.computation(),
                                            &log_sizes,
                                        )
                                        .unwrap_or(f64::INFINITY);
                                        if std::env::var("H_DEBUG").is_ok() {
                                            eprintln!("[dbg]   cand{ci} dflops={:.3}", f2 - base_flops);
                                        }
                                        priced.push((f2 - base_flops, ci, act));
                                    }
                                }
                              }
                            }
                        }
                        priced.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                        for (_, ci, act) in priced.into_iter().take(8) {
                            if let Ok(s2) = transition::apply(&state, act.clone()) {
                                let c = score(s2.computation());
                                let better = match &best {
                                    None => true,
                                    Some((b, _, _)) => c < *b,
                                };
                                if better {
                                    best = Some((c, act, format!("biclique def{d} cand{ci}")));
                                }
                            }
                        }
                    }
                    if dbg {
                        eprintln!("[dbg] def{d} biclique phase: {:?}", dstart.elapsed());
                    }
                    // permutation
                    if let Ok(ActionSpace::Permutation(space)) =
                        action::query(&state, ActionQuery::PermutationFactor(action::DefinitionPosition(d)))
                    {
                        let snaps = space.snapshot();
                        for ci in 0..snaps.len() {
                            let (_, roots, uses) = &snaps[ci];
                            let rmask = vec![true; roots.len()];
                            let umask = vec![true; uses.len()];
                            if let Ok(act) = space.select(ci, &rmask, &umask) {
                                if let Ok(s2) = transition::apply(&state, act.clone()) {
                                    let c = score(s2.computation());
                                    let better = match &best {
                                        None => true,
                                        Some((b, _, _)) => c < *b,
                                    };
                                    if better {
                                        best = Some((c, act, format!("permute def{d} cand{ci}")));
                                    }
                                }
                            }
                        }
                    }
                    if dbg {
                        eprintln!("[dbg] def{d} permutation phase: {:?}", dstart.elapsed());
                    }
                    // parenthesize: balanced splits of multi-factor terms
                    let nterms = state.computation().definitions[d].rhs.len();
                    for tpos in 0..nterms {
                        let nf = state.computation().definitions[d].rhs[tpos].factors.len();
                        if nf < 3 {
                            continue;
                        }
                        if let Ok(ActionSpace::Parenthesize(space)) =
                            action::query(&state, ActionQuery::Parenthesize(action::TermPosition {
                                definition: action::DefinitionPosition(d),
                                term: tpos,
                            }))
                        {
                            for cut in 1..nf {
                                let mask: Vec<bool> =
                                    (0..nf).map(|i| i < cut).collect();
                                if let Ok(act) = space.select(&mask) {
                                    if let Ok(s2) = transition::apply(&state, act.clone()) {
                                        let c = score(s2.computation());
                                        if dbg {
                                            eprintln!(
                                                "[dbg]   paren def{d} term{tpos} cut{cut}: {c:.6} s"
                                            );
                                        }
                                        let better = match &best {
                                            None => true,
                                            Some((b, _, _)) => c < *b,
                                        };
                                        if better {
                                            best = Some((
                                                c,
                                                act,
                                                format!("paren def{d} term{tpos} cut{cut}"),
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if dbg {
                    eprintln!("[dbg] step {step} total: {:?}", step_start.elapsed());
                }
                let mut best = best.map(|(c, act, desc)| (c, vec![act], desc));
                if lookahead {
                    let base_flops = gristmill_rewrites::cost::log_flops(
                        state.computation(),
                        &log_sizes,
                    )
                    .unwrap_or(f64::INFINITY);
                    let mut firsts: Vec<(f64, action::Action, String)> = Vec::new();
                    for d in 0..state.computation().definitions.len() {
                        for tpos in 0..state.computation().definitions[d].rhs.len() {
                            let nf = state.computation().definitions[d].rhs[tpos].factors.len();
                            if nf < 3 {
                                continue;
                            }
                            if let Ok(ActionSpace::Parenthesize(space)) = action::query(
                                &state,
                                ActionQuery::Parenthesize(action::TermPosition {
                                    definition: action::DefinitionPosition(d),
                                    term: tpos,
                                }),
                            ) {
                                for cut in 1..nf {
                                    let mask: Vec<bool> = (0..nf).map(|i| i < cut).collect();
                                    if let Ok(act) = space.select(&mask) {
                                        if let Ok(s1) = transition::apply(&state, act.clone()) {
                                            let f1 = gristmill_rewrites::cost::log_flops(
                                                s1.computation(),
                                                &log_sizes,
                                            )
                                            .unwrap_or(f64::INFINITY);
                                            firsts.push((
                                                f1 - base_flops,
                                                act,
                                                format!("paren def{d} term{tpos} cut{cut}"),
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    firsts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                    for (_, first, first_desc) in firsts.into_iter().take(4) {
                        if let Ok(s1) = transition::apply(&state, first.clone()) {
                            if let Some((c, second, second_desc)) =
                                best_kernel_action(&s1, &score, &log_sizes, dbg)
                            {
                                let better = match &best {
                                    None => true,
                                    Some((b, _, _)) => c < *b,
                                };
                                if better {
                                    best = Some((
                                        c,
                                        vec![first, second],
                                        format!("lookahead {first_desc} => {second_desc}"),
                                    ));
                                }
                            }
                        }
                    }
                }
                let accepted = match &best {
                    Some((c, _, _)) => {
                        if score_mode == "flops" {
                            // log_flops is additive: 0.5% relative FLOP
                            // improvement is an absolute log difference.
                            *c < cur - (1.0 + accept).ln()
                        } else {
                            *c < cur * (1.0 - accept)
                        }
                    }
                    None => false,
                };
                match best {
                    Some((c, actions, desc)) if accepted => {
                        eprintln!("[select] step {step}: {desc}  {cur:.4} -> {c:.4} s");
                        trace.push(serde_json::json!({
                            "step": step, "action": desc,
                            "before_s": cur, "after_s": c,
                        }));
                        for act in actions {
                            state = transition::apply(&state, act)
                                .map_err(|e| anyhow::anyhow!("apply: {e:?}"))?;
                        }
                        cur = c;
                    }
                    _ => {
                        eprintln!("[select] converged at step {step}, predicted {cur:.4} s");
                        break;
                    }
                }
            }
            // Final: convert, verify exactly, optionally save.
            let prog = bridge::computation_to_program(
                state.computation(),
                &input_ids,
                &output_ids,
            );
            let search_log_flops = truthmill::log_flops(&problem, &prog)
                .map_err(|e| anyhow::anyhow!("search flops: {e:?}"))?;
            let (prog, log_flops, flop_fallback) = enforce_flop_cap(
                &problem,
                prog,
                &baseline_program,
                flop_cap_log,
            )?;
            if flop_fallback {
                eprintln!("[select] Axis-1 cap missed; using verified baseline fallback");
            }
            let ok = truthmill::verify(&problem, &prog)
                .map_err(|e| anyhow::anyhow!("verify error: {e:?}"))?;
            eprintln!("[select] truthmill::verify(final): {ok}");
            if !ok {
                anyhow::bail!("final program failed exact verification");
            }
            if let Some(path) = &out {
                std::fs::write(path, serde_json::to_string(&prog)?)?;
                eprintln!("[select] wrote {}", path.display());
            }
            let record = serde_json::json!({
                "kind": "select",
                "score": score_mode,
                "engine": engine,
                "problem": name,
                "sizes": sizes_map(&problem),
                "threads": threads,
                "predicted_s": if flop_fallback { None } else { Some(cur) },
                "log10_flops": log_flops / std::f64::consts::LN_10,
                "search_log10_flops": search_log_flops / std::f64::consts::LN_10,
                "flop_cap_log10": flop_cap_log / std::f64::consts::LN_10,
                "flop_fallback": flop_fallback,
                "baseline": baseline_path.display().to_string(),
                "trace": trace,
            });
            println!("{}", serde_json::to_string(&record)?);
        }
        Cmd::Profile {
            problem,
            sizes,
            threads,
            iters,
            seed,
            transform,
            candidate,
            layout,
        } => {
            let name = stem(&problem);
            let problem = load_problem(&problem, &sizes)?;
            if dense::estimate_input_bytes(&problem) / (1 << 20) > 3000 {
                anyhow::bail!("memory guard: inputs too large; shrink sizes");
            }
            let program = load_program(&problem, &transform, 256, candidate)?;
            let orig = dense::generate_inputs_for(&problem, &[&program], seed);
            let (program, blocks) = apply_layout_with(&problem, program, &layout, orig,
                &layout2::PolicyParams::default(), threads)?;
            let tensor_blocks = interp::to_tensor_blocks_permuted(&blocks.permuted)?;
            let est = interp::estimate_memory(&program, &tensor_blocks, &problem.sizes, 1 << 30);
            if !est.risky_ops.is_empty() || est.peak_live_bytes / (1 << 20) > 6000 {
                anyhow::bail!("memory guard tripped: {:?}", est.risky_ops);
            }
            let mut backend = CpuBackend::with_threads(threads).map_err(anyhow::Error::msg)?;
            let profiles = interp::profile_ops(&program, &tensor_blocks, &mut backend, iters, &problem.sizes)?;
            let total: f64 = profiles.iter().map(|p| p.median_ms).sum();
            let mut out = Vec::new();
            for p in &profiles {
                out.push(serde_json::json!({
                    "value": p.value,
                    "kind": p.kind,
                    "median_ms": p.median_ms,
                    "union_log10_flops": p.union_log10_flops,
                }));
            }
            let record = serde_json::json!({
                "kind": "profile",
                "transform": transform,
                "layout": layout,
                "problem": name,
                "sizes": sizes_map(&problem),
                "threads": threads,
                "total_median_ms": total,
                "ops": out,
            });
            println!("{}", serde_json::to_string_pretty(&record)?);
        }
        Cmd::Report {
            problem,
            sizes,
            threads,
            iters,
            warmup,
            seed,
            transform,
            candidate,
            layout,
            penalty,
            store,
            anchor,
            top,
            log,
        } => {
            cmd_report(
                problem, sizes, threads, iters, warmup, seed, transform, candidate, layout,
                penalty, store, anchor, top, log,
            )?;
        }
    }
    Ok(())
}

pub struct LayoutResult {
    pub orig: dense::Blocks,
    pub permuted: layout::PermutedBlocks,
    pub layouts: layout::Layouts,
}

fn apply_layout_with(
    problem: &Problem,
    program: truthmill::Program,
    layout: &str,
    orig: dense::Blocks,
    params: &layout2::PolicyParams,
    threads: usize,
) -> Result<(truthmill::Program, LayoutResult)> {
    match layout {
        "none" => {
            let permuted = layout::apply_layouts_to_blocks(&orig, &layout::Layouts::new());
            Ok((
                program,
                LayoutResult {
                    orig,
                    permuted,
                    layouts: layout::Layouts::new(),
                },
            ))
        }
        "greedy-layout" => {
            let layouts = layout::plan_layouts(problem, &program);
            let program = layout::apply_layouts_to_program(&program, &layouts);
            let permuted = layout::apply_layouts_to_blocks(&orig, &layouts);
            Ok((
                program,
                LayoutResult {
                    orig,
                    permuted,
                    layouts,
                },
            ))
        }
        "model" => {
            let rates = costmodel::rates_for_threads(threads);
            let (blocks, values) = layout3::plan_layouts_model(problem, &program, &rates, 3);
            let program = layout::apply_layouts_to_program_v2(&program, &values);
            let permuted = layout::apply_layouts_to_blocks(&orig, &blocks);
            Ok((
                program,
                LayoutResult {
                    orig,
                    permuted,
                    layouts: blocks,
                },
            ))
        }
        "v2" => {
            let (blocks, values) = layout2::plan_layouts_v2(problem, &program, params);
            if std::env::var("H_DEBUG").is_ok() {
                eprintln!("value perms: {values:?}");
                for (id, def) in program.values.iter().enumerate() {
                    match def {
                        truthmill::ValueDef::Einsum(op) => eprintln!(
                            "v{id} E out={:?} in={:?} args={:?}",
                            op.code.output, op.code.inputs, op.args
                        ),
                        truthmill::ValueDef::LinComb(op) => eprintln!(
                            "v{id} L out={:?} in={:?} args={:?}",
                            op.code.output, op.code.inputs, op.args
                        ),
                    }
                }
            }
            let program = layout::apply_layouts_to_program_v2(&program, &values);
            let permuted = layout::apply_layouts_to_blocks(&orig, &blocks);
            Ok((
                program,
                LayoutResult {
                    orig,
                    permuted,
                    layouts: blocks,
                },
            ))
        }
        other => anyhow::bail!("unknown layout: {other}"),
    }
}

fn enforce_flop_cap(
    problem: &Problem,
    program: truthmill::Program,
    baseline: &truthmill::Program,
    cap_log: f64,
) -> Result<(truthmill::Program, f64, bool)> {
    let log_flops = truthmill::log_flops(problem, &program)
        .map_err(|e| anyhow::anyhow!("candidate flops: {e:?}"))?;
    if log_flops <= cap_log + 1e-10 {
        return Ok((program, log_flops, false));
    }
    let baseline_log = truthmill::log_flops(problem, baseline)
        .map_err(|e| anyhow::anyhow!("baseline flops: {e:?}"))?;
    anyhow::ensure!(baseline_log <= cap_log + 1e-10, "baseline exceeds its own cap");
    Ok((baseline.clone(), baseline_log, true))
}

#[cfg(test)]
mod axis1_tests {
    use super::*;

    #[test]
    fn selector_split_matches_default_kernel_envelope() {
        let cli = Cli::try_parse_from(["harness", "select", "problem.json"]).unwrap();
        match cli.cmd {
            Cmd::Select { split_above, .. } => assert_eq!(split_above, 20),
            _ => panic!("expected select command"),
        }
    }

    #[test]
    fn over_cap_mp3_schedule_falls_back_to_baseline() {
        let mut problem = Problem::from_json(include_str!(
            "../../../benches/chemistry/mp3.json"
        ))
        .unwrap();
        problem.sizes.insert(DomainId(0), 16);
        problem.sizes.insert(DomainId(1), 64);
        let selected: truthmill::Program = serde_json::from_str(include_str!(
            "../../../benches/candidates/mp3_selected.json"
        ))
        .unwrap();
        let baseline: truthmill::Program = serde_json::from_str(include_str!(
            "../../../benches/candidates/mp3_gristmill.json"
        ))
        .unwrap();
        let baseline_log = truthmill::log_flops(&problem, &baseline).unwrap();
        let cap = baseline_log + 1.05f64.ln();
        let selected_log = truthmill::log_flops(&problem, &selected).unwrap();
        assert!(selected_log > cap);
        let (fallback, fallback_log, fell_back) =
            enforce_flop_cap(&problem, selected, &baseline, cap).unwrap();
        assert!(fell_back);
        assert_eq!(fallback, baseline);
        assert!(fallback_log <= cap);
        let (_, baseline_log, fell_back) =
            enforce_flop_cap(&problem, baseline.clone(), &baseline, cap).unwrap();
        assert!(!fell_back);
        assert!(baseline_log <= cap);
    }
}

fn load_program(
    problem: &Problem,
    transform: &str,
    factmem_threshold_mb: u64,
    candidate: Option<PathBuf>,
) -> Result<truthmill::Program> {
    if let Some(path) = candidate {
        let text = std::fs::read_to_string(&path)?;
        let program: truthmill::Program =
            serde_json::from_str(&text).map_err(anyhow::Error::msg)?;
        let ok = truthmill::verify(problem, &program)
            .map_err(|e| anyhow::anyhow!("verify error: {e:?}"))?;
        anyhow::ensure!(ok, "candidate program failed truthmill::verify");
        eprintln!("candidate verified: {}", path.display());
        return Ok(program);
    }
    apply_transform(problem, transform, factmem_threshold_mb)
}

fn apply_transform(
    problem: &Problem,
    transform: &str,
    factmem_threshold_mb: u64,
) -> Result<truthmill::Program> {
    match transform {
        "none" => Ok(problem.reference.clone()),
        "greedy" => {
            let program = transform::greedy_factor(&problem.reference, &problem.sizes);
            let ok = truthmill::verify(problem, &program)
                .map_err(|e| anyhow::anyhow!("verify error: {e:?}"))?;
            anyhow::ensure!(ok, "truthmill::verify rejected greedy transform");
            Ok(program)
        }
        "factmem" => {
            let program = transform::factor_with_threshold(
                &problem.reference,
                &problem.sizes,
                (factmem_threshold_mb as u128) << 20,
            );
            let ok = truthmill::verify(problem, &program)
                .map_err(|e| anyhow::anyhow!("verify error: {e:?}"))?;
            anyhow::ensure!(ok, "truthmill::verify rejected factmem transform");
            Ok(program)
        }
        "cse" => {
            let program = transform::greedy_factor(&problem.reference, &problem.sizes);
            let program = transform::cse_exact(&program);
            let ok = truthmill::verify(problem, &program)
                .map_err(|e| anyhow::anyhow!("verify error: {e:?}"))?;
            anyhow::ensure!(ok, "truthmill::verify rejected cse transform");
            Ok(program)
        }
        "cse-only" => {
            let program = transform::cse_exact(&problem.reference);
            let ok = truthmill::verify(problem, &program)
                .map_err(|e| anyhow::anyhow!("verify error: {e:?}"))?;
            anyhow::ensure!(ok, "truthmill::verify rejected cse-only transform");
            Ok(program)
        }
        "share" | "sharecse" => {
            let program = transform::share_pairs(
                &problem.reference,
                &problem.sizes,
                100,
                256 * (1 << 20),
            );
            let program = if transform == "sharecse" {
                transform::cse_exact(&program)
            } else {
                program
            };
            let ok = truthmill::verify(problem, &program)
                .map_err(|e| anyhow::anyhow!("verify error: {e:?}"))?;
            anyhow::ensure!(ok, "truthmill::verify rejected share transform");
            Ok(program)
        }
        other => anyhow::bail!("unknown transform: {other}"),
    }
}

fn spearman(rows: &[(f64, f64)]) -> f64 {
    fn ranks(xs: &[f64]) -> Vec<f64> {
        let mut idx: Vec<usize> = (0..xs.len()).collect();
        idx.sort_by(|&a, &b| xs[a].partial_cmp(&xs[b]).unwrap());
        let mut r = vec![0.0; xs.len()];
        for (rank, &i) in idx.iter().enumerate() {
            r[i] = rank as f64;
        }
        r
    }
    let a: Vec<f64> = rows.iter().map(|r| r.0).collect();
    let b: Vec<f64> = rows.iter().map(|r| r.1).collect();
    let (ra, rb) = (ranks(&a), ranks(&b));
    let n = ra.len() as f64;
    let (ma, mb) = (ra.iter().sum::<f64>() / n, rb.iter().sum::<f64>() / n);
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for i in 0..ra.len() {
        cov += (ra[i] - ma) * (rb[i] - mb);
        va += (ra[i] - ma).powi(2);
        vb += (rb[i] - mb).powi(2);
    }
    cov / (va * vb).sqrt().max(1e-300)
}

fn problem_name(problem: &Problem) -> String {
    format!("{:x}", fxhash(problem))
}

fn fxhash(problem: &Problem) -> u64 {
    let json = problem.to_json().unwrap_or_default();
    let mut hash = 0xcbf29ce484222325u64;
    for b in json.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn sizes_map(problem: &Problem) -> BTreeMap<u32, u64> {
    problem
        .sizes
        .iter()
        .map(|(d, s)| (d.0, *s))
        .collect()
}

fn optimize_large_definitions(
    computation: &gristmill_rewrites::repr::Computation,
    outputs: &[gristmill_rewrites::repr::TensorId],
    score: &dyn Fn(&gristmill_rewrites::repr::Computation) -> f64,
    max_steps: usize,
    accept: f64,
) -> Result<gristmill_rewrites::repr::Computation> {
    use gristmill_rewrites::state::State;
    use gristmill_rewrites::verify::equivalent_states;

    let mut comp = computation.clone();
    let mut state = State::new(comp.clone(), outputs.to_vec())
        .map_err(|e| anyhow::anyhow!("hybrid state: {e:?}"))?;
    let mut cur = score(&comp);
    for step in 0..max_steps {
        let proposals = propose::propose_above(&comp, 21);
        let mut best: Option<(f64, usize)> = None;
        for (pi, proposal) in proposals.iter().enumerate() {
            let comp2 = propose::apply_proposal(&comp, proposal);
            let c = score(&comp2);
            let better = match &best {
                None => true,
                Some((b, _)) => c < *b,
            };
            if better {
                best = Some((c, pi));
            }
        }
        match best {
            Some((c, pi)) if c < cur * (1.0 - accept) => {
                let comp2 = propose::apply_proposal(&comp, &proposals[pi]);
                let state2 = State::new(comp2.clone(), outputs.to_vec())
                    .map_err(|e| anyhow::anyhow!("hybrid state: {e:?}"))?;
                let equiv = equivalent_states(&state, &state2)
                    .map_err(|e| anyhow::anyhow!("hybrid verify: {e:?}"))?;
                if !equiv {
                    eprintln!("[hybrid] step {step}: structural proposal failed equivalence");
                    break;
                }
                eprintln!(
                    "[hybrid] step {step}: group{} free{:?} {cur:.4} -> {c:.4} s",
                    proposals[pi].group_size, proposals[pi].free_domains
                );
                comp = comp2;
                state = state2;
                cur = c;
            }
            _ => break,
        }
    }
    Ok(comp)
}

#[allow(clippy::too_many_arguments)]
fn run_structural(
    problem: &Problem,
    name: &str,
    input_ids: &[u32],
    output_ids: &[u32],
    outputs: &[gristmill_rewrites::repr::TensorId],
    rates: &costmodel::CostRates,
    threads: usize,
    max_steps: usize,
    accept: f64,
    out: Option<PathBuf>,
    baseline: &truthmill::Program,
    flop_cap_log: f64,
) -> Result<()> {
    use gristmill_rewrites::state::State;
    use gristmill_rewrites::verify::equivalent_states;

    let (mut comp, _) = bridge::problem_to_computation(problem, 1000);
    bridge::split_large_definitions(&mut comp, 12);
    let dbg_p = std::env::var("H_DEBUG").is_ok();
    let price = |comp: &gristmill_rewrites::repr::Computation| -> f64 {
        let t0 = std::time::Instant::now();
        let prog = bridge::computation_to_program(comp, input_ids, output_ids);
        let t1 = std::time::Instant::now();
        let (bl, vl) = layout3::plan_layouts_model(problem, &prog, rates, 1);
        let t2 = std::time::Instant::now();
        let prog = layout::apply_layouts_to_program_v2(&prog, &vl);
        let c: f64 = costmodel::program_costs(&prog, &problem.sizes, &bl, rates)
            .iter()
            .sum();
        if dbg_p {
            eprintln!("    [price] bridge {:?} layout {:?} cost {:?}", t1-t0, t2-t1, t2.elapsed());
        }
        c
    };
    let t_start = std::time::Instant::now();
    let mut cur = price(&comp);
    let mut state = State::new(comp.clone(), outputs.to_vec())
        .map_err(|e| anyhow::anyhow!("state: {e:?}"))?;
    eprintln!("[structural] initial predicted {cur:.4} s");
    let mut trace = Vec::new();
    let mut rejected: Vec<(usize, Vec<u32>)> = Vec::new();
    for step in 0..max_steps {
        let proposals = propose::propose(&comp);
        eprintln!("[structural] step {step}: {} proposals, {} defs",
            proposals.len(), comp.definitions.len());
        if proposals.is_empty() {
            eprintln!("[structural] no more proposals at step {step}");
            break;
        }
        let mut best: Option<(f64, usize)> = None;
        for (pi, p) in proposals.iter().enumerate() {
            if rejected
                .iter()
                .any(|(g, f)| *g == p.group_size && *f == p.free_domains)
            {
                continue; // class failed verification before; skip
            }
            eprintln!("[structural]   prop{pi}: group{} free{:?} ...", p.group_size, p.free_domains);
            let t0 = std::time::Instant::now();
            let comp2 = propose::apply_proposal(&comp, p);
            eprintln!("[structural]   prop{pi}: applied, {} defs", comp2.definitions.len());
            let t1 = std::time::Instant::now();
            let c = price(&comp2);
            eprintln!("[structural]   prop{pi}: apply {:?} price {:?} -> {c:.4}",
                t1 - t0, t1.elapsed());
            let better = match &best {
                None => true,
                Some((b, _)) => c < *b,
            };
            if better {
                best = Some((c, pi));
            }
        }
        match best {
            Some((c, pi)) if c < cur * (1.0 - accept) => {
                let comp2 = propose::apply_proposal(&comp, &proposals[pi]);
                // Gate: exact equivalence via the kernel.
                let state2 = State::new(comp2.clone(), outputs.to_vec())
                    .map_err(|e| anyhow::anyhow!("state: {e:?}"))?;
                let equiv = equivalent_states(&state, &state2)
                    .map_err(|e| anyhow::anyhow!("verify error: {e:?}"))?;
                if !equiv {
                    eprintln!(
                        "[structural] step {step}: proposal group{} free{:?} FAILED equivalence; blocklisted",
                        proposals[pi].group_size, proposals[pi].free_domains
                    );
                    rejected.push((proposals[pi].group_size, proposals[pi].free_domains.clone()));
                    trace.push(serde_json::json!({
                        "step": step, "event": "verify_reject",
                        "group": proposals[pi].group_size,
                        "free_domains": proposals[pi].free_domains,
                    }));
                    continue; // try next step with fresh proposals
                }
                eprintln!(
                    "[structural] step {step}: group{} free{:?}  {cur:.4} -> {c:.4} s",
                    proposals[pi].group_size, proposals[pi].free_domains
                );
                trace.push(serde_json::json!({
                    "step": step, "group": proposals[pi].group_size,
                    "free_domains": proposals[pi].free_domains,
                    "before_s": cur, "after_s": c,
                }));
                comp = comp2;
                state = state2;
                cur = c;
            }
            _ => {
                eprintln!("[structural] converged at step {step}, predicted {cur:.4} s");
                break;
            }
        }
    }
    let prog = bridge::computation_to_program(&comp, input_ids, output_ids);
    let (prog, log_flops, flop_fallback) =
        enforce_flop_cap(problem, prog, baseline, flop_cap_log)?;
    if flop_fallback {
        eprintln!("[structural] Axis-1 cap missed; using verified baseline fallback");
    }
    if std::env::var("H_DEBUG").is_ok() {
        std::fs::write("/tmp/dbg_program.json", serde_json::to_string_pretty(&prog)?)?;
    }
    let ok = truthmill::verify(problem, &prog).map_err(|e| anyhow::anyhow!("verify: {e:?}"))?;
    eprintln!("[structural] truthmill::verify(final): {ok}");
    if !ok {
        anyhow::bail!("final program failed exact verification");
    }
    if let Some(path) = &out {
        std::fs::write(path, serde_json::to_string(&prog)?)?;
        eprintln!("[structural] wrote {}", path.display());
    }
    let record = serde_json::json!({
        "kind": "select", "engine": "structural", "problem": name,
        "sizes": sizes_map(problem), "threads": threads,
        "predicted_s": if flop_fallback { None } else { Some(cur) },
        "log10_flops": log_flops / std::f64::consts::LN_10,
        "flop_cap_log10": flop_cap_log / std::f64::consts::LN_10,
        "flop_fallback": flop_fallback,
        "search_wall_s": t_start.elapsed().as_secs_f64(),
        "trace": trace,
    });
    println!("{}", serde_json::to_string(&record)?);
    Ok(())
}

// ---------------------------------------------------------------------------
// report: the decomposed run panel (M1.0)
// ---------------------------------------------------------------------------

fn load_avg() -> Option<f64> {
    let s = std::fs::read_to_string("/proc/loadavg").ok()?;
    s.split_whitespace().next()?.parse().ok()
}

fn git_hash() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn sha256_file(path: &PathBuf) -> String {
    // Cheap digest: length + FNV-1a of contents (not cryptographic; detects edits).
    use std::hash::Hasher;
    match std::fs::read(path) {
        Ok(bytes) => {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            std::hash::Hash::hash(&bytes, &mut h);
            format!("{:016x}:{}", h.finish(), bytes.len())
        }
        Err(_) => "unreadable".into(),
    }
}

#[allow(clippy::too_many_arguments)]
fn cmd_report(
    problem: PathBuf,
    sizes: Option<String>,
    threads: usize,
    iters: usize,
    warmup: usize,
    seed: u64,
    transform: String,
    candidate: Option<PathBuf>,
    layout: String,
    penalty: f64,
    store: Option<PathBuf>,
    anchor: Option<PathBuf>,
    top: usize,
    log: Option<PathBuf>,
) -> Result<()> {
    let load = load_avg();
    let git = git_hash();
    let name = stem(&problem);
    let cand_label = candidate
        .as_ref()
        .map(|p| stem(p))
        .unwrap_or_else(|| format!("transform:{transform}"));
    let problem = load_problem(&problem, &sizes)?;
    if dense::estimate_input_bytes(&problem) / (1 << 20) > 3000 {
        anyhow::bail!("memory guard: inputs too large; shrink sizes");
    }
    let program = load_program(&problem, &transform, 256, candidate.clone())?;
    let orig = dense::generate_inputs_for(&problem, &[&program], seed);
    let (program, blocks) = apply_layout_with(
        &problem,
        program,
        &layout,
        orig,
        &layout2::PolicyParams {
            p_big: penalty,
            p_small: penalty,
            thresh: 1e8,
        },
        threads,
    )?;
    let tensor_blocks = interp::to_tensor_blocks_permuted(&blocks.permuted)?;
    let est = interp::estimate_memory(&program, &tensor_blocks, &problem.sizes, 1 << 30);
    let peak_mb = est.peak_live_bytes / (1 << 20);
    if !est.risky_ops.is_empty() || peak_mb > 6000 {
        anyhow::bail!("memory guard tripped: {:?}", est.risky_ops);
    }

    let rates = costmodel::rates_for_threads(threads);
    let predicted = costmodel::program_costs(&program, &problem.sizes, &blocks.layouts, &rates);
    let mut backend = CpuBackend::with_threads(threads).map_err(anyhow::Error::msg)?;

    // Whole-program steady-state timing (the number that counts).
    let plans = interp::prepare(&program, &tensor_blocks)?;
    let mut times = Vec::with_capacity(iters);
    let mut csum = 0.0;
    for i in 0..(warmup + iters) {
        let start = Instant::now();
        let outputs = interp::run_prepared(&program, &plans, &tensor_blocks, &mut backend)?;
        if i >= warmup {
            times.push(start.elapsed().as_secs_f64() * 1e3);
        }
        csum += interp::checksum(&outputs);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let total_ms = times[times.len() / 2];

    // Per-op profile (op-level medians; sum may differ from total due to
    // whole-program cache/liveness state — report both, never conflate).
    let profiles = interp::profile_ops(&program, &tensor_blocks, &mut backend, iters, &problem.sizes)?;
    let opsum_ms: f64 = profiles.iter().map(|p| p.median_ms).sum();

    // Category totals (einsum vs lincomb).
    let mut cat: BTreeMap<&str, f64> = BTreeMap::new();
    for p in &profiles {
        *cat.entry(p.kind).or_default() += p.median_ms;
    }

    // Panel.
    println!("=== report: {name} / {cand_label} ===");
    println!(
        "sizes {:?}  threads {threads}  total {:.1} ms  (sum of op medians {:.1} ms)",
        sizes_map(&problem),
        total_ms,
        opsum_ms
    );
    println!("peak-memory estimate: {peak_mb} MB");
    println!();
    println!("top ops by measured time (vs model):");
    let mut rows: Vec<(usize, &interp::OpProfile, f64)> = profiles
        .iter()
        .enumerate()
        .map(|(i, p)| (i, p, predicted.get(p.value).copied().unwrap_or(f64::NAN) * 1e3))
        .collect();
    rows.sort_by(|a, b| b.1.median_ms.partial_cmp(&a.1.median_ms).unwrap());
    for (i, p, pred_ms) in rows.iter().take(top) {
        let ratio = p.median_ms / pred_ms.max(1e-9);
        let flag = if ratio > 2.0 || ratio < 0.5 {
            "  MODEL-MISS"
        } else {
            ""
        };
        println!(
            "  op#{i:<4} {:<8} {:9.2} ms ({:4.1}%)  model {:9.2} ms{flag}",
            p.kind,
            p.median_ms,
            100.0 * p.median_ms / opsum_ms.max(1e-9),
            pred_ms
        );
    }
    let rest: f64 = rows.iter().skip(top).map(|r| r.1.median_ms).sum();
    println!("  ... +{} ops aggregated: {:.1} ms", rows.len().saturating_sub(top), rest);
    println!();
    println!("model-miss outliers (|measured/model| outside [0.5, 2], any size):");
    let mut n_miss = 0;
    for (i, p, pred_ms) in &rows {
        let ratio = p.median_ms / pred_ms.max(1e-9);
        if ratio > 2.0 || ratio < 0.5 {
            println!(
                "  op#{i:<4} {:<8} measured {:8.2} ms  model {:8.2} ms  ratio {:5.2}x",
                p.kind, p.median_ms, pred_ms, ratio
            );
            n_miss += 1;
        }
    }
    if n_miss == 0 {
        println!("  (none)");
    }
    println!();
    println!("category totals:");
    for (k, v) in &cat {
        println!("  {k:<10} {:9.2} ms ({:4.1}%)", v, 100.0 * v / opsum_ms.max(1e-9));
    }
    let top1 = rows.first().map(|r| r.1.median_ms / opsum_ms.max(1e-9)).unwrap_or(0.0);
    let top5: f64 = rows.iter().take(5).map(|r| r.1.median_ms).sum::<f64>() / opsum_ms.max(1e-9);
    println!();
    println!("concentration: top-1 op {:.1}%, top-5 ops {:.1}%", 100.0 * top1, 100.0 * top5);

    // Anchor comparison (aggregates only: no op-level alignment exists).
    if let Some(anchor_path) = &anchor {
        let txt = std::fs::read_to_string(anchor_path)?;
        let a: serde_json::Value = serde_json::from_str(&txt)?;
        let a_total = a["total_ms"].as_f64().unwrap_or(f64::NAN);
        println!();
        println!("vs anchor {} (pinned):", anchor_path.display());
        println!("  total: {total_ms:.1} ms vs {a_total:.1} ms  ({:.3}x)", total_ms / a_total);
        if let Some(obj) = a["categories"].as_object() {
            let a_opsum = a["opsum_ms"].as_f64().unwrap_or(f64::NAN);
            for (k, v) in obj {
                let share_a = v.as_f64().unwrap_or(0.0) / a_opsum;
                let share_c = cat.get(k.as_str()).copied().unwrap_or(0.0) / opsum_ms.max(1e-9);
                println!(
                    "  {k:<10} share: ours {:.1}% vs anchor {:.1}%",
                    100.0 * share_c,
                    100.0 * share_a
                );
            }
        }
        println!(
            "  anchor provenance: git {}  date {}  load {}  digest {}",
            a["git"].as_str().unwrap_or("?"),
            a["date"].as_str().unwrap_or("?"),
            a["load"].as_f64().unwrap_or(f64::NAN),
            a["digest"].as_str().unwrap_or("?")
        );
        println!("  (drift check: re-store the anchor if machine/backend changed)");
    }

    // Validity footer.
    let protocol_ok = iters >= 10 && warmup >= 3;
    println!();
    println!(
        "validity: iters {iters} warmup {warmup} {}  load {:.2}  git {}  checksum {:.3}",
        if protocol_ok { "protocol-OK" } else { "SUB-PROTOCOL (need >=10/>=3)" },
        load.unwrap_or(f64::NAN),
        git,
        csum
    );

    // Store / log.
    let rec = serde_json::json!({
        "kind": "report",
        "problem": name,
        "candidate": cand_label,
        "digest": candidate.as_ref().map(sha256_file).unwrap_or_default(),
        "sizes": sizes_map(&problem),
        "threads": threads,
        "iters": iters,
        "warmup": warmup,
        "protocol_ok": protocol_ok,
        "total_ms": total_ms,
        "opsum_ms": opsum_ms,
        "peak_mb": peak_mb,
        "categories": cat,
        "top1_share": top1,
        "top5_share": top5,
        "load": load,
        "git": git,
        "checksum": csum,
        "date": chrono_lite_now(),
        "ops": profiles.iter().enumerate().map(|(i, p)| serde_json::json!({
            "value": i, "kind": p.kind, "median_ms": p.median_ms,
            "model_ms": predicted.get(p.value).copied().unwrap_or(f64::NAN) * 1e3,
        })).collect::<Vec<_>>(),
    });
    if let Some(path) = &store {
        std::fs::write(path, serde_json::to_string_pretty(&rec)?)?;
        eprintln!("anchor stored: {}", path.display());
    }
    if let Some(path) = &log {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
        writeln!(f, "{}", serde_json::to_string(&rec)?)?;
    }
    Ok(())
}

fn chrono_lite_now() -> String {
    // Avoid a chrono dependency: seconds since epoch is enough for ordering.
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("unix:{secs}")
}
