//! Tenferro-backed interpreter for truthmill programs.
//!
//! Executes the SSA program op by op with einsum plans (prepared once for the
//! iterative regime) plus dense add/mul for linear combinations and scalar
//! coefficients. This is the naive baseline executor; optimized schedules
//! reuse the same machinery with a transformed program.

use std::collections::HashMap;
use std::hint::black_box;

use anyhow::Result;
use tenferro_cpu::CpuBackend;
use tenferro_einsum::{ConcreteEinsumPlan, EinsumSubscripts, TensorEinsumExt};
use tenferro_runtime::TensorSessionOpsExt;
use tenferro_tensor::Tensor;
use truthmill::{Code, Program, ValueDef, ValueRef};

use crate::dense::Blocks;

fn coeff_f64(c: &truthmill::Coefficient) -> f64 {
    *c.numer() as f64 / *c.denom() as f64
}

/// Input blocks converted to tenferro tensors, each with its storage
/// permutation (identity for unpermuted blocks). Storage axis k holds the
/// logical axis `perm[k]`.
pub type TensorBlocks = HashMap<(u32, Vec<u32>), (Tensor, Vec<usize>)>;

pub fn to_tensor_blocks(blocks: &Blocks) -> Result<TensorBlocks> {
    let mut out = HashMap::new();
    for (key, dense) in blocks {
        let tensor = Tensor::from_vec_col_major(dense.dims.clone(), dense.data.clone())
            .map_err(anyhow::Error::msg)?;
        let perm: Vec<usize> = (0..dense.dims.len()).collect();
        out.insert(key.clone(), (tensor, perm));
    }
    Ok(out)
}

pub fn to_tensor_blocks_permuted(
    blocks: &crate::layout::PermutedBlocks,
) -> Result<TensorBlocks> {
    let mut out = HashMap::new();
    for (key, (dense, perm)) in blocks {
        let tensor = Tensor::from_vec_col_major(dense.dims.clone(), dense.data.clone())
            .map_err(anyhow::Error::msg)?;
        out.insert(key.clone(), (tensor, perm.clone()));
    }
    Ok(out)
}

fn subscripts(code: &Code) -> (Vec<Vec<u32>>, Vec<u32>) {
    let inputs: Vec<Vec<u32>> = code
        .inputs
        .iter()
        .map(|labels| labels.iter().map(|l| l.0).collect())
        .collect();
    let output: Vec<u32> = code.output.iter().map(|l| l.0).collect();
    (inputs, output)
}

/// Argument labels in physical storage order: storage axis k holds logical
/// axis `perm[k]`, and logical axis a carries label `labels[a]`.
fn physical_labels(labels: &[truthmill::IndexId], perm: Option<&[usize]>) -> Vec<u32> {
    match perm {
        Some(p) => p.iter().map(|&k| labels[k].0).collect(),
        None => labels.iter().map(|l| l.0).collect(),
    }
}

pub enum OpPlan {
    Einsum {
        plan: ConcreteEinsumPlan,
        scale: Option<Tensor>,
    },
    LinComb {
        // Per-arg permutation plan (identity included; plans own the copy).
        permutes: Vec<ConcreteEinsumPlan>,
        coeffs: Vec<f64>,
    },
}

fn resolve<'a>(
    blocks: &'a TensorBlocks,
    values: &'a [Tensor],
    reference: ValueRef,
    labels: &[truthmill::IndexId],
    code: &Code,
) -> Result<(&'a Tensor, Option<&'a [usize]>)> {
    match reference {
        ValueRef::Tensor(tensor) => {
            let pattern: Vec<u32> = labels.iter().map(|l| code.domains[l].0).collect();
            let (block, perm) = blocks
                .get(&(tensor.0, pattern.clone()))
                .ok_or_else(|| anyhow::anyhow!("missing block {} {:?}", tensor.0, pattern))?;
            let identity = perm.iter().enumerate().all(|(k, &p)| k == p);
            Ok((block, if identity { None } else { Some(perm) }))
        }
        ValueRef::Value(id) => Ok((&values[id.0 as usize], None)),
    }
}


/// Prepare all einsum plans once (iterative regime), evaluating shapes with a
/// scratch single-thread backend.
pub fn prepare(program: &Program, blocks: &TensorBlocks) -> Result<Vec<OpPlan>> {
    let mut values: Vec<Tensor> = Vec::new();
    let mut backend = CpuBackend::with_threads(1).map_err(anyhow::Error::msg)?;
    let mut ops = Vec::new();
    for definition in &program.values {
        match definition {
            ValueDef::Einsum(op) => {
                let (_, output) = subscripts(&op.code);
                let mut tensors: Vec<&Tensor> = Vec::new();
                let mut phys_inputs: Vec<Vec<u32>> = Vec::new();
                for (a, labels) in op.args.iter().zip(&op.code.inputs) {
                    let (tensor, perm) = resolve(blocks, &values, *a, labels, &op.code)?;
                    tensors.push(tensor);
                    phys_inputs.push(physical_labels(labels, perm));
                }
                let input_refs: Vec<&[u32]> = phys_inputs.iter().map(Vec::as_slice).collect();
                let subs = EinsumSubscripts::new(&input_refs, &output);
                let plan = ConcreteEinsumPlan::prepare_subscripts(&tensors, &subs)
                    .map_err(anyhow::Error::msg)?;
                let coeff = coeff_f64(&op.coeff);
                let scale = if (coeff - 1.0).abs() > 1e-15 {
                    Some(Tensor::from_vec_col_major(vec![], vec![coeff]).map_err(anyhow::Error::msg)?)
                } else {
                    None
                };
                let value = plan.execute(&tensors, &mut backend).map_err(anyhow::Error::msg)?;
                values.push(value);
                ops.push(OpPlan::Einsum { plan, scale });
            }
            ValueDef::LinComb(op) => {
                let (_inputs, output) = subscripts(&op.code);
                let mut permutes = Vec::new();
                let mut shape_value: Option<Tensor> = None;
                for (arg, labels) in op.args.iter().zip(&op.code.inputs) {
                    let (tensor, perm) = resolve(blocks, &values, *arg, labels, &op.code)?;
                    let in_labels = physical_labels(labels, perm);
                    let subs = EinsumSubscripts::new(&[&in_labels[..]], &output);
                    let plan = ConcreteEinsumPlan::prepare_subscripts(&[tensor], &subs)
                        .map_err(anyhow::Error::msg)?;
                    if shape_value.is_none() {
                        shape_value = Some(
                            plan.execute(&[tensor], &mut backend).map_err(anyhow::Error::msg)?,
                        );
                    }
                    permutes.push(plan);
                }
                values.push(shape_value.expect("LinComb with no args"));
                ops.push(OpPlan::LinComb {
                    permutes,
                    coeffs: op.coeff.iter().map(coeff_f64).collect(),
                });
            }
        }
    }
    Ok(ops)
}

/// Execute one full iteration of `program` using prepared plans.
pub fn run_prepared(
    program: &Program,
    plans: &[OpPlan],
    blocks: &TensorBlocks,
    backend: &mut CpuBackend,
) -> Result<Vec<Tensor>> {
    let mut values: Vec<Tensor> = Vec::with_capacity(program.values.len());
    for (definition, plan) in program.values.iter().zip(plans) {
        match (definition, plan) {
            (
                ValueDef::Einsum(op),
                OpPlan::Einsum { plan, scale },
            ) => {
                let mut tensors: Vec<&Tensor> = Vec::new();
                let mut phys_inputs: Vec<Vec<u32>> = Vec::new();
                for (a, labels) in op.args.iter().zip(&op.code.inputs) {
                    let (tensor, perm) = resolve(blocks, &values, *a, labels, &op.code)?;
                    tensors.push(tensor);
                    phys_inputs.push(physical_labels(labels, perm));
                }
                let mut value = plan
                    .execute(&tensors, backend)
                    .map_err(anyhow::Error::msg)?;
                if let Some(scalar) = scale {
                    value = value
                        .mul(scalar, backend)
                        .map_err(anyhow::Error::msg)?;
                }
                values.push(value);
            }
            (
                ValueDef::LinComb(op),
                OpPlan::LinComb { permutes, coeffs },
            ) => {
                let mut acc: Option<Tensor> = None;
                for ((arg, labels), (perm, coeff)) in op
                    .args
                    .iter()
                    .zip(&op.code.inputs)
                    .zip(permutes.iter().zip(coeffs))
                {
                    let (tensor, _) = resolve(blocks, &values, *arg, labels, &op.code)?;
                    let mut term = perm
                        .execute(&[tensor], backend)
                        .map_err(anyhow::Error::msg)?;
                    if (*coeff - 1.0).abs() > 1e-15 {
                        let scalar = Tensor::from_vec_col_major(vec![], vec![*coeff])
                            .map_err(anyhow::Error::msg)?;
                        term = term.mul(&scalar, backend).map_err(anyhow::Error::msg)?;
                    }
                    acc = Some(match acc {
                        None => term,
                        Some(a) => a.add(&term, backend).map_err(anyhow::Error::msg)?,
                    });
                }
                values.push(acc.expect("LinComb with no args"));
            }
            _ => anyhow::bail!("plan/program mismatch"),
        }
    }
    let mut values: Vec<Option<Tensor>> = values.into_iter().map(Some).collect();
    Ok(program
        .outputs
        .iter()
        .map(|o| values[o.0 as usize].take().expect("output"))
        .collect())
}

/// Interpret one full iteration without any plan reuse (cold path).
pub fn run_cold(
    program: &Program,
    blocks: &TensorBlocks,
    backend: &mut CpuBackend,
) -> Result<Vec<Tensor>> {
    let mut values: Vec<Tensor> = Vec::with_capacity(program.values.len());
    for definition in &program.values {
        match definition {
            ValueDef::Einsum(op) => {
                let (_, output) = subscripts(&op.code);
                let mut tensors: Vec<&Tensor> = Vec::new();
                let mut phys_inputs: Vec<Vec<u32>> = Vec::new();
                for (a, labels) in op.args.iter().zip(&op.code.inputs) {
                    let (tensor, perm) = resolve(blocks, &values, *a, labels, &op.code)?;
                    tensors.push(tensor);
                    phys_inputs.push(physical_labels(labels, perm));
                }
                let input_refs: Vec<&[u32]> = phys_inputs.iter().map(Vec::as_slice).collect();
                let subs = EinsumSubscripts::new(&input_refs, &output);
                let mut value = tensors
                    .as_slice()
                    .einsum_subscripts(&subs, backend)
                    .map_err(anyhow::Error::msg)?;
                let coeff = coeff_f64(&op.coeff);
                if (coeff - 1.0).abs() > 1e-15 {
                    let scalar = Tensor::from_vec_col_major(vec![], vec![coeff])
                        .map_err(anyhow::Error::msg)?;
                    value = value.mul(&scalar, backend).map_err(anyhow::Error::msg)?;
                }
                values.push(value);
            }
            ValueDef::LinComb(op) => {
                let (_, output) = subscripts(&op.code);
                let mut acc: Option<Tensor> = None;
                for (coeff, (arg, labels)) in
                    op.coeff.iter().zip(op.args.iter().zip(&op.code.inputs))
                {
                    let (tensor, perm) = resolve(blocks, &values, *arg, labels, &op.code)?;
                    let in_labels = physical_labels(labels, perm);
                    let subs = EinsumSubscripts::new(&[&in_labels[..]], &output);
                    let mut term = [tensor]
                        .as_slice()
                        .einsum_subscripts(&subs, backend)
                        .map_err(anyhow::Error::msg)?;
                    let coeff = coeff_f64(coeff);
                    if (coeff - 1.0).abs() > 1e-15 {
                        let scalar = Tensor::from_vec_col_major(vec![], vec![coeff])
                            .map_err(anyhow::Error::msg)?;
                        term = term.mul(&scalar, backend).map_err(anyhow::Error::msg)?;
                    }
                    acc = Some(match acc {
                        None => term,
                        Some(a) => a.add(&term, backend).map_err(anyhow::Error::msg)?,
                    });
                }
                values.push(acc.expect("LinComb with no args"));
            }
        }
    }
    let mut values: Vec<Option<Tensor>> = values.into_iter().map(Some).collect();
    Ok(program
        .outputs
        .iter()
        .map(|o| values[o.0 as usize].take().expect("output"))
        .collect())
}

/// Checksum over outputs to keep timing loops honest.
pub fn checksum(outputs: &[Tensor]) -> f64 {
    let mut sum = 0.0;
    for tensor in outputs {
        let data = tensor.as_slice::<f64>().expect("f64 tensor");
        // Sample at most 1024 entries to keep the checksum cheap.
        let stride = (data.len() / 1024).max(1);
        let mut acc = 0.0;
        for v in data.iter().step_by(stride) {
            acc += v.abs();
        }
        sum += acc + data.len() as f64;
    }
    black_box(sum)
}

pub struct OpProfile {
    pub value: usize,
    pub kind: &'static str,
    pub median_ms: f64,
    pub union_log10_flops: f64,
}

/// Time each SSA op individually (median over `iters` executions of that op),
/// using prepared plans. Warmup: one full program pass plus one per-op call.
pub fn profile_ops(
    program: &Program,
    blocks: &TensorBlocks,
    backend: &mut CpuBackend,
    iters: usize,
    sizes: &std::collections::BTreeMap<truthmill::DomainId, u64>,
) -> Result<Vec<OpProfile>> {
    let plans = prepare(program, blocks)?;
    // One full pass to warm caches and fill `values`.
    let mut values: Vec<Tensor> = Vec::with_capacity(program.values.len());
    for (definition, plan) in program.values.iter().zip(&plans) {
        let value = exec_one(definition, plan, blocks, &values, backend)?;
        values.push(value);
    }
    let mut profiles = Vec::new();
    for (n, (definition, plan)) in program.values.iter().zip(&plans).enumerate() {
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let start = std::time::Instant::now();
            let value = exec_one(definition, plan, blocks, &values, backend)?;
            std::hint::black_box(&value);
            times.push(start.elapsed().as_secs_f64() * 1e3);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let union_log10_flops = match definition {
            ValueDef::Einsum(op) => {
                let mut labels: Vec<u32> = op.code.output.iter().map(|l| l.0).collect();
                for input in &op.code.inputs {
                    for l in input {
                        if !labels.contains(&l.0) {
                            labels.push(l.0);
                        }
                    }
                }
                let log10: f64 = labels
                    .iter()
                    .map(|l| (sizes[&op.code.domains[&truthmill::IndexId(*l)]] as f64).log10())
                    .sum();
                log10
            }
            ValueDef::LinComb(_) => 0.0,
        };
        let kind = match definition {
            ValueDef::Einsum(_) => "einsum",
            ValueDef::LinComb(_) => "lincomb",
        };
        profiles.push(OpProfile {
            value: n,
            kind,
            median_ms: times[times.len() / 2],
            union_log10_flops,
        });
    }
    Ok(profiles)
}

fn exec_one(
    definition: &ValueDef,
    plan: &OpPlan,
    blocks: &TensorBlocks,
    values: &[Tensor],
    backend: &mut CpuBackend,
) -> Result<Tensor> {
    match (definition, plan) {
        (ValueDef::Einsum(op), OpPlan::Einsum { plan, scale }) => {
            let mut tensors: Vec<&Tensor> = Vec::new();
            for (a, labels) in op.args.iter().zip(&op.code.inputs) {
                let (tensor, _) = resolve(blocks, values, *a, labels, &op.code)?;
                tensors.push(tensor);
            }
            let mut value = plan.execute(&tensors, backend).map_err(anyhow::Error::msg)?;
            if let Some(scalar) = scale {
                value = value.mul(scalar, backend).map_err(anyhow::Error::msg)?;
            }
            Ok(value)
        }
        (ValueDef::LinComb(op), OpPlan::LinComb { permutes, coeffs }) => {
            let mut acc: Option<Tensor> = None;
            for ((arg, labels), (perm, coeff)) in op
                .args
                .iter()
                .zip(&op.code.inputs)
                .zip(permutes.iter().zip(coeffs))
            {
                let (tensor, _) = resolve(blocks, values, *arg, labels, &op.code)?;
                let mut term = perm.execute(&[tensor], backend).map_err(anyhow::Error::msg)?;
                if (*coeff - 1.0).abs() > 1e-15 {
                    let scalar = Tensor::from_vec_col_major(vec![], vec![*coeff])
                        .map_err(anyhow::Error::msg)?;
                    term = term.mul(&scalar, backend).map_err(anyhow::Error::msg)?;
                }
                acc = Some(match acc {
                    None => term,
                    Some(a) => a.add(&term, backend).map_err(anyhow::Error::msg)?,
                });
            }
            Ok(acc.expect("LinComb with no args"))
        }
        _ => anyhow::bail!("plan/program mismatch"),
    }
}
