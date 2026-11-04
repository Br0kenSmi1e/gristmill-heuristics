//! Calibration suite (M1.1, F-004/F-016): measured achieved-rate grid over
//! einsum op classes, producing a versioned rates file for cost model v2.
//!
//! The current model prices ops with a binary clean/slow GEMM
//! classification (60.4/22.5 GF/s t1). F-004 recorded rank-6
//! intermediate-forming ops at ~0.6 GF/s — a third regime the model cannot
//! see. This suite measures achieved GF/s across the discriminating axes:
//! operand count (binary vs n-ary), output rank (scalar reduction vs
//! high-rank intermediate), and operand stride state, at several sizes and
//! thread counts. Every run passes the F-006 load guard (in main.rs).

use std::collections::BTreeMap;
use std::time::Instant;

use anyhow::{Context, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tenferro_cpu::CpuBackend;
use tenferro_einsum::TensorEinsumExt;
use tenferro_tensor::Tensor;

/// One calibration case: an einsum over letters, with per-letter sizes
/// derived from the (o, v) chemistry scale pair.
struct Case {
    name: &'static str,
    einsum: &'static str,
    /// letter -> size rule: O (occupied), V (virtual), or Fixed.
    dims: &'static [(char, Dim)],
    /// (o, v) scale points to measure at.
    scales: &'static [(usize, usize)],
    /// FLOP model: Contraction = 2*prod(all extents); Elementwise = prod.
    flop_model: FlopModel,
}

#[derive(Clone, Copy)]
enum Dim {
    O,
    V,
    Fixed(usize),
}

#[derive(Clone, Copy)]
enum FlopModel {
    Contraction,
    Elementwise,
}

use Dim::{Fixed, O, V};

/// Reconstruct one case at one scale: (physical input label orders, output
/// labels, per-label extents). Label id = char as u32 (consistent within a
/// case; only relative order matters for the model).
fn case_spec(
    case: &Case,
    o: usize,
    v: usize,
) -> Result<(Vec<Vec<u32>>, Vec<u32>, BTreeMap<u32, usize>)> {
    let (inputs, output) = parse_einsum(case.einsum)?;
    let mut size_of = BTreeMap::new();
    for (c, d) in case.dims {
        let s = match d {
            O => o,
            V => v,
            Fixed(n) => *n,
        };
        size_of.insert(*c as u32, s);
    }
    let phys: Vec<Vec<u32>> = inputs
        .iter()
        .map(|inp| inp.iter().map(|c| *c as u32).collect())
        .collect();
    let out: Vec<u32> = output.iter().map(|c| *c as u32).collect();
    Ok((phys, out, size_of))
}

/// The grid. Anchors first (continuity with the v1 committed rates), then
/// the F-004 discriminators.
const CASES: &[Case] = &[
    Case {
        name: "gemm_clean",
        einsum: "mk,kn->mn",
        dims: &[('m', Fixed(1024)), ('k', Fixed(1024)), ('n', Fixed(1024))],
        scales: &[(0, 0)],
        flop_model: FlopModel::Contraction,
    },
    Case {
        name: "ccd_slowpath_anchor",
        einsum: "abij,cdab->cdij",
        dims: &[
            ('a', V), ('b', V), ('i', O), ('j', O),
            ('c', V), ('d', V),
        ],
        scales: &[(16, 64)],
        flop_model: FlopModel::Contraction,
    },
    // F-004 discriminators: is the hole about operand count, output rank,
    // or stride state?
    Case {
        name: "bin_r6_scalar",
        einsum: "abcdef,abcdef->",
        dims: &[('a', V), ('b', V), ('c', V), ('d', O), ('e', O), ('f', O)],
        scales: &[(4, 8), (8, 16), (12, 24)],
        flop_model: FlopModel::Contraction,
    },
    Case {
        name: "bin_r6_scalar_strided",
        einsum: "abcdef,fedcba->",
        dims: &[('a', V), ('b', V), ('c', V), ('d', O), ('e', O), ('f', O)],
        scales: &[(4, 8), (8, 16), (12, 24)],
        flop_model: FlopModel::Contraction,
    },
    Case {
        name: "bin_r6_to_r6",
        einsum: "abcd,cefg->dabfge",
        dims: &[
            ('a', V), ('b', V), ('c', V), ('d', O),
            ('e', V), ('f', O), ('g', O),
        ],
        scales: &[(4, 8), (8, 16)],
        flop_model: FlopModel::Contraction,
    },
    Case {
        name: "tern_r6_scalar",
        einsum: "abcdef,efbdac,fabdec->",
        dims: &[('a', V), ('b', V), ('c', V), ('d', O), ('e', O), ('f', O)],
        scales: &[(4, 8), (8, 16), (12, 24)],
        flop_model: FlopModel::Contraction,
    },
    Case {
        name: "quad_r6_scalar",
        einsum: "abcd,efgd,cefgab,gefbac->",
        dims: &[
            ('a', O), ('b', O), ('c', V), ('d', O),
            ('e', V), ('f', V), ('g', O),
        ],
        scales: &[(4, 8), (8, 16), (12, 24)],
        flop_model: FlopModel::Contraction,
    },
    Case {
        name: "dot_r2_scalar",
        einsum: "a,a->",
        dims: &[('a', V)],
        scales: &[(0, 1048576)],
        flop_model: FlopModel::Contraction,
    },
    Case {
        name: "bin_r4_scalar",
        einsum: "abcd,abcd->",
        dims: &[('a', V), ('b', V), ('c', O), ('d', O)],
        scales: &[(8, 16), (16, 32)],
        flop_model: FlopModel::Contraction,
    },
    Case {
        name: "bin_r6_out1",
        einsum: "abcdef,abcdef->a",
        dims: &[('a', V), ('b', V), ('c', V), ('d', O), ('e', O), ('f', O)],
        scales: &[(4, 8), (8, 16), (12, 24)],
        flop_model: FlopModel::Contraction,
    },
    // The same ternary contraction as a manual binary chain: permute +
    // elementwise multiply, then binary full reduction. Discriminates
    // "n-ary lowering is the hole" from "rank-6 ops are the hole".
    Case {
        name: "tern_r6_chain_step1_mul",
        einsum: "abcdef,efbdac->abcdef",
        dims: &[('a', V), ('b', V), ('c', V), ('d', O), ('e', O), ('f', O)],
        scales: &[(4, 8), (8, 16), (12, 24)],
        flop_model: FlopModel::Elementwise,
    },
    Case {
        name: "tern_r6_chain_step2_reduce",
        einsum: "abcdef,fabdec->",
        dims: &[('a', V), ('b', V), ('c', V), ('d', O), ('e', O), ('f', O)],
        scales: &[(4, 8), (8, 16), (12, 24)],
        flop_model: FlopModel::Contraction,
    },
    // Low-rank n-ary control: a 3-operand chain that stays low-rank.
    Case {
        name: "tern_lowrank_control",
        einsum: "ab,bc,cd->ad",
        dims: &[('a', V), ('b', V), ('c', V), ('d', V)],
        scales: &[(0, 64), (0, 128)],
        flop_model: FlopModel::Contraction,
    },
    // F-016 gap classes (surfaced on ccsd_gristmill op#19/op#23):
    Case {
        // k-heavy low-rank-output: contract v^2, output v x v.
        name: "kheavy_r2out",
        einsum: "ab,bcad->cd",
        dims: &[('a', V), ('b', V), ('c', V), ('d', V)],
        scales: &[(8, 16), (16, 32), (20, 40), (24, 48)],
        flop_model: FlopModel::Contraction,
    },
    Case {
        // Single contracted index, rank-4 output (op#23 class).
        name: "singlek_r4out",
        einsum: "ab,cdae->bcde",
        dims: &[('a', V), ('b', O), ('c', V), ('d', V), ('e', V)],
        scales: &[(8, 16), (12, 24), (16, 32), (20, 40)],
        flop_model: FlopModel::Contraction,
    },
    Case {
        // Pure rank-6 permute: proxy for the per-input permute floor that
        // rank-6 LinComb ops showed (F-016 class c).
        name: "permute_r6",
        einsum: "abcdef->fedcba",
        dims: &[('a', V), ('b', V), ('c', V), ('d', O), ('e', O), ('f', O)],
        scales: &[(4, 8), (12, 24)],
        flop_model: FlopModel::Elementwise,
    },
    Case {
        // Rank-4 permute control (isolates the rank dependence of the
        // permute floor).
        name: "permute_r4",
        einsum: "abcd->dcba",
        dims: &[('a', V), ('b', V), ('c', O), ('d', O)],
        scales: &[(8, 32), (16, 64)],
        flop_model: FlopModel::Elementwise,
    },
];

fn rand_tensor(dims: &[usize], seed: u64) -> Tensor {
    let len: usize = dims.iter().product();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let data: Vec<f64> = (0..len).map(|_| rng.random::<f64>() - 0.5).collect();
    Tensor::from_vec_col_major(dims.to_vec(), data).unwrap()
}

/// Parse "in0,in1,...->out" and build (input label letters, output letters).
fn parse_einsum(spec: &str) -> Result<(Vec<Vec<char>>, Vec<char>)> {
    let (lhs, rhs) = spec
        .split_once("->")
        .with_context(|| format!("bad einsum spec: {spec}"))?;
    let inputs = lhs.split(',').map(|s| s.chars().collect()).collect();
    let output = rhs.chars().collect();
    Ok((inputs, output))
}

fn time_median<F: FnMut(&mut CpuBackend)>(mut f: F, backend: &mut CpuBackend, warmup: usize, iters: usize) -> (f64, f64, f64) {
    for _ in 0..warmup {
        f(backend);
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        f(backend);
        times.push(start.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (times[times.len() / 2], times[0], times[times.len() - 1])
}

pub fn run(
    threads_list: &[usize],
    iters: usize,
    warmup: usize,
    load: &crate::LoadInfo,
    out: Option<&std::path::PathBuf>,
) -> Result<()> {
    let mut records = Vec::new();
    for case in CASES {
        let (inputs, output) = parse_einsum(case.einsum)?;
        for &(o, v) in case.scales {
            let size_of: BTreeMap<char, usize> = case
                .dims
                .iter()
                .map(|(c, d)| {
                    let s = match d {
                        O => o,
                        V => v,
                        Fixed(n) => *n,
                    };
                    (*c, s)
                })
                .collect();
            // Sanity: every letter in the einsum has a size.
            for c in inputs.iter().flatten().chain(output.iter()) {
                anyhow::ensure!(
                    size_of.contains_key(c),
                    "case {}: letter {c} has no size rule",
                    case.name
                );
            }
            let extents = || -> usize { size_of.values().product() };
            let flops = match case.flop_model {
                FlopModel::Contraction => 2.0 * extents() as f64,
                FlopModel::Elementwise => extents() as f64,
            };
            let bytes: f64 = inputs
                .iter()
                .map(|inp| inp.iter().map(|c| size_of[c]).product::<usize>() as f64 * 8.0)
                .sum::<f64>()
                + output.iter().map(|c| size_of[c]).product::<usize>() as f64 * 8.0;
            let tensors: Vec<Tensor> = inputs
                .iter()
                .enumerate()
                .map(|(i, inp)| {
                    rand_tensor(
                        &inp.iter().map(|c| size_of[c]).collect::<Vec<_>>(),
                        (i + 1) as u64,
                    )
                })
                .collect();
            for &threads in threads_list {
                let mut backend =
                    CpuBackend::with_threads(threads).map_err(anyhow::Error::msg)?;
                let refs: Vec<&Tensor> = tensors.iter().collect();
                let (median, min, max) = time_median(
                    |b| {
                        let r = refs.as_slice().einsum(case.einsum, b).unwrap();
                        std::hint::black_box(r);
                    },
                    &mut backend,
                    warmup,
                    iters,
                );
                let rec = serde_json::json!({
                    "case": case.name,
                    "einsum": case.einsum,
                    "o": o,
                    "v": v,
                    "threads": threads,
                    "median_ms": median,
                    "min_ms": min,
                    "max_ms": max,
                    "flops": flops,
                    "gflops": flops / median / 1e6,
                    "gbytes_s": bytes / (median / 1e3) / 1e9,
                    "n_operands": inputs.len(),
                    "out_rank": output.len(),
                });
                println!("{}", serde_json::to_string(&rec)?);
                records.push(rec);
            }
        }
    }
    let doc = serde_json::json!({
        "kind": "calibrate",
        "schema": 1,
        "git": crate::git_hash(),
        "date": crate::chrono_lite_now(),
        "load": load.l1,
        "load5": load.l5,
        "load15": load.l15,
        "max_load": load.max_load,
        "load_override": load.override_used,
        "iters": iters,
        "warmup": warmup,
        "records": records,
    });
    if let Some(path) = out {
        std::fs::write(path, serde_json::to_string_pretty(&doc)?)?;
        eprintln!("calibration written: {}", path.display());
    }
    Ok(())
}

/// One model-vs-measured comparison row.
pub struct ModelRatio {
    pub case: String,
    pub o: usize,
    pub v: usize,
    pub threads: usize,
    pub measured_ms: f64,
    pub model_ms: f64,
    /// measured / model; 1.0 is perfect, >1 means the model under-predicts.
    pub ratio: f64,
}

/// Documented exceptions to the 2x gate. Each entry MUST carry a reason;
/// adding an exception is a ledger-visible act (cite the F-entry).
const MODEL_CI_EXCEPTIONS: &[(&str, usize, &str)] = &[
    (
        "bin_r6_out1",
        16,
        "rank-1-output reduction at t16 is pathological AND unstable (2.6x \
         session variance, F-004); priced pessimistically, error is an \
         over-prediction at small sizes (safe direction for selection)",
    ),
    (
        "tern_lowrank_control",
        0,
        "small-m GEMM chains: per-step dispatch overhead dominates at \
         v<=128; pre-existing v1 gap, absolute cost tiny (F-016 data)",
    ),
    (
        "quad_r6_scalar",
        16,
        "t16 timing unstable across guard-compliant sessions (61.6 vs 15.9 \
         ms at (8,16): 3.9x apart, trials/m11-calibration.jsonl); the model \
         sits between the two sessions, so no point rate passes both",
    ),
];

/// Price every record of a calibration document with the current cost
/// model. Pure function of the committed measurements: no timing, so it
/// runs in CI and cargo tests.
pub fn model_ratios(doc: &serde_json::Value) -> Result<Vec<ModelRatio>> {
    let mut rows = Vec::new();
    for rec in doc["records"]
        .as_array()
        .context("calibration doc missing records")?
    {
        let name = rec["case"].as_str().context("record missing case")?;
        let case = CASES
            .iter()
            .find(|c| c.name == name)
            .with_context(|| format!("unknown calibration case {name}"))?;
        let o = rec["o"].as_u64().unwrap_or(0) as usize;
        let v = rec["v"].as_u64().unwrap_or(0) as usize;
        let threads = rec["threads"].as_u64().unwrap_or(1) as usize;
        let measured_ms = rec["median_ms"].as_f64().context("missing median_ms")?;
        let (phys, out, size_of) = case_spec(case, o, v)?;
        let rates = crate::costmodel::rates_for_threads(threads);
        let model_ms = crate::costmodel::einsum_cost_model(&phys, &out, &size_of, &rates)
            .max(rates.dispatch)
            * 1e3;
        rows.push(ModelRatio {
            case: name.to_string(),
            o,
            v,
            threads,
            measured_ms,
            model_ms,
            ratio: measured_ms / model_ms,
        });
    }
    Ok(rows)
}

/// The 2x gate: returns the list of NON-excepted failures.
pub fn model_ci_failures(rows: &[ModelRatio]) -> Vec<String> {
    let mut failures = Vec::new();
    for r in rows {
        let ok = (0.5..=2.0).contains(&r.ratio);
        if ok {
            continue;
        }
        let excepted = MODEL_CI_EXCEPTIONS
            .iter()
            .any(|(name, threads, _)| *name == r.case && (*threads == 0 || *threads == r.threads));
        if !excepted {
            failures.push(format!(
                "{} o={} v={} t{}: measured {:.2} ms vs model {:.2} ms ({:.2}x)",
                r.case, r.o, r.v, r.threads, r.measured_ms, r.model_ms, r.ratio
            ));
        }
    }
    failures
}

#[cfg(test)]
mod tests {
    /// Model-vs-calibration CI gate (F-004 acceptance): every calibration
    /// case within 2x, exceptions documented in MODEL_CI_EXCEPTIONS.
    #[test]
    fn model_v2_within_2x_of_calibration_grid() {
        let doc: serde_json::Value = serde_json::from_str(include_str!(
            "../../../profiles/calibrate-v1.json"
        ))
        .unwrap();
        let rows = super::model_ratios(&doc).unwrap();
        assert!(!rows.is_empty());
        let failures = super::model_ci_failures(&rows);
        assert!(failures.is_empty(), "model CI failures:\n{}", failures.join("\n"));
    }
}
