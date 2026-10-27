//! Harness: correctness oracle and steady-state timing for truthmill problems
//! executed on tenferro.

mod dense;
mod interp;
mod layout;
mod layout2;
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
    /// Print symbolic FLOP terms for reference and greedy-factored programs.
    Flops {
        problem: PathBuf,
        /// Evaluate polynomials numerically at these sizes, e.g. "0=100,1=1000".
        #[arg(long)]
        sizes: Option<String>,
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
                &layout2::PolicyParams::default())?;
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
            let (program, blocks) = apply_layout_with(&problem, program, &layout, orig, &params)?;
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
            let (program, blocks) = apply_layout(&problem, program, &layout, seed,
                &layout2::PolicyParams::default())?;
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
}

fn apply_layout(
    problem: &Problem,
    program: truthmill::Program,
    layout: &str,
    seed: u64,
    params: &layout2::PolicyParams,
) -> Result<(truthmill::Program, LayoutResult)> {
    let orig = dense::generate_inputs(problem, seed);
    apply_layout_with(problem, program, layout, orig, params)
}

fn apply_layout_with(
    problem: &Problem,
    program: truthmill::Program,
    layout: &str,
    orig: dense::Blocks,
    params: &layout2::PolicyParams,
) -> Result<(truthmill::Program, LayoutResult)> {
    match layout {
        "none" => {
            let permuted = layout::apply_layouts_to_blocks(&orig, &layout::Layouts::new());
            Ok((program, LayoutResult { orig, permuted }))
        }
        "greedy-layout" => {
            let layouts = layout::plan_layouts(problem, &program);
            let program = layout::apply_layouts_to_program(&program, &layouts);
            let permuted = layout::apply_layouts_to_blocks(&orig, &layouts);
            Ok((program, LayoutResult { orig, permuted }))
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
            Ok((program, LayoutResult { orig, permuted }))
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
        other => anyhow::bail!("unknown transform: {other}"),
    }
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
