//! Harness: correctness oracle and steady-state timing for truthmill problems
//! executed on tenferro.

mod bridge;
mod costmodel;
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
        #[arg(long, default_value = "none")]
        layout: String,
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
        /// Layout planning: none|greedy-layout
        #[arg(long, default_value = "none")]
        layout: String,
    },
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
            layout,
        } => {
            let name = stem(&problem);
            let problem = load_problem(&problem, &sizes)?;
            if dense::estimate_input_bytes(&problem) / (1 << 20) > 3000 {
                anyhow::bail!("memory guard: inputs too large; shrink sizes");
            }
            let program = apply_transform(&problem, &transform, 256)?;
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
        Cmd::Select {
            problem,
            sizes,
            threads,
            max_steps,
            out,
            accept,
        } => {
            let name = stem(&problem);
            let problem = load_problem(&problem, &sizes)?;
            let input_ids: Vec<u32> = dense::collect_patterns(&problem)
                .keys()
                .map(|t| t.0)
                .collect();
            let (computation, outputs) = bridge::problem_to_computation(&problem, 1000);
            let output_ids: Vec<u32> = outputs.iter().map(|t| t.0).collect();
            let rates = costmodel::rates_for_threads(threads);

            let score = |comp: &gristmill_rewrites::repr::Computation| -> f64 {
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
                        let log_sizes: std::collections::BTreeMap<
                            gristmill_rewrites::repr::RangeId,
                            f64,
                        > = problem
                            .sizes
                            .iter()
                            .map(|(dom, s)| {
                                (gristmill_rewrites::repr::RangeId(dom.0), (*s as f64).ln())
                            })
                            .collect();
                        let base_flops = gristmill_rewrites::cost::log_flops(
                            state.computation(),
                            &log_sizes,
                        )
                        .unwrap_or(f64::INFINITY);
                        let mut priced: Vec<(f64, usize, action::Action)> = Vec::new();
                        for ci in 0..space.candidate_count() {
                            if let Some((nl, nr)) = space.shape(ci) {
                                let left = vec![true; nl];
                                let right = vec![true; nr];
                                if let Ok(act) = space.select(ci, &left, &right) {
                                    if let Ok(s2) = transition::apply(&state, act.clone()) {
                                        let f2 = gristmill_rewrites::cost::log_flops(
                                            s2.computation(),
                                            &log_sizes,
                                        )
                                        .unwrap_or(f64::INFINITY);
                                        priced.push((f2 - base_flops, ci, act));
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
                match best {
                    Some((c, act, desc)) if c < cur * (1.0 - accept) => {
                        eprintln!("[select] step {step}: {desc}  {cur:.4} -> {c:.4} s");
                        trace.push(serde_json::json!({
                            "step": step, "action": desc,
                            "before_s": cur, "after_s": c,
                        }));
                        state = transition::apply(&state, act)
                            .map_err(|e| anyhow::anyhow!("apply: {e:?}"))?;
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
                "problem": name,
                "sizes": sizes_map(&problem),
                "threads": threads,
                "predicted_s": cur,
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
            layout,
        } => {
            let name = stem(&problem);
            let problem = load_problem(&problem, &sizes)?;
            if dense::estimate_input_bytes(&problem) / (1 << 20) > 3000 {
                anyhow::bail!("memory guard: inputs too large; shrink sizes");
            }
            let program = apply_transform(&problem, &transform, 256)?;
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
