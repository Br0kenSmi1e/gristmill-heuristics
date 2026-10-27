//! Harness: correctness oracle and steady-state timing for truthmill problems
//! executed on tenferro.

mod dense;
mod interp;
mod oracle;

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
        /// Use the cold interpreter (replan every call) instead of prepared plans.
        #[arg(long)]
        cold: bool,
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

fn load_problem(path: &PathBuf, sizes: &Option<String>) -> Result<Problem> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {path:?}"))?;
    let mut problem = Problem::from_json(&text).map_err(anyhow::Error::msg)?;
    parse_sizes(&mut problem, sizes)?;
    truthmill::validate(&problem).map_err(|e| anyhow::anyhow!("validation: {e:?}"))?;
    Ok(problem)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Check {
            problem,
            sizes,
            seed,
        } => {
            let problem = load_problem(&problem, &sizes)?;
            let blocks = dense::generate_inputs(&problem, seed);
            let expected = oracle::eval_program(&problem, &problem.reference, &blocks)?;
            let tensor_blocks = interp::to_tensor_blocks(&blocks)?;
            let mut backend = CpuBackend::with_threads(1).map_err(anyhow::Error::msg)?;
            let actual = interp::run_cold(&problem.reference, &tensor_blocks, &mut backend)?;
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
                "problem": problem_name(&problem),
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
            cold,
        } => {
            let problem = load_problem(&problem, &sizes)?;
            let blocks = dense::generate_inputs(&problem, seed);
            let tensor_blocks = interp::to_tensor_blocks(&blocks)?;
            let mut backend =
                CpuBackend::with_threads(threads).map_err(anyhow::Error::msg)?;
            let log_flops = truthmill::log_flops(&problem, &problem.reference)
                .map_err(|e| anyhow::anyhow!("cost: {e:?}"))?;
            let plans = if cold {
                None
            } else {
                Some(interp::prepare(&problem.reference, &tensor_blocks)?)
            };
            let mut times = Vec::with_capacity(iters);
            let mut sum = 0.0;
            for i in 0..(warmup + iters) {
                let start = Instant::now();
                let outputs = match &plans {
                    Some(plans) => interp::run_prepared(
                        &problem.reference,
                        plans,
                        &tensor_blocks,
                        &mut backend,
                    )?,
                    None => interp::run_cold(&problem.reference, &tensor_blocks, &mut backend)?,
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
                "problem": problem_name(&problem),
                "sizes": sizes_map(&problem),
                "threads": threads,
                "mode": if cold { "cold" } else { "prepared" },
                "iters": iters,
                "warmup": warmup,
                "median_ms": median,
                "min_ms": times[0],
                "max_ms": times[times.len() - 1],
                "log10_flops": log_flops,
                "checksum": sum,
            });
            println!("{}", serde_json::to_string(&record)?);
        }
    }
    Ok(())
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
