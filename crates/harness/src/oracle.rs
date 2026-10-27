//! Independent scalar-loop evaluator for truthmill programs.
//!
//! This path shares no code with the tenferro interpreter: it evaluates the
//! SSA semantics directly with explicit index loops, and is only meant for
//! tiny sizes where the brute-force cost is irrelevant.

use anyhow::{anyhow, Result};
use truthmill::{Code, Problem, Program, ValueDef, ValueRef};

use crate::dense::{Blocks, Dense};

fn coeff_f64(c: &truthmill::Coefficient) -> f64 {
    *c.numer() as f64 / *c.denom() as f64
}

fn block_of<'a>(blocks: &'a Blocks, reference: ValueRef, labels: &[truthmill::IndexId], code: &Code, values: &'a [Dense]) -> Result<DenseView<'a>> {
    match reference {
        ValueRef::Tensor(tensor) => {
            let pattern: Vec<u32> = labels.iter().map(|l| code.domains[l].0).collect();
            blocks
                .get(&(tensor.0, pattern.clone()))
                .map(DenseView::Borrowed)
                .ok_or_else(|| anyhow!("missing block for tensor {} pattern {:?}", tensor.0, pattern))
        }
        ValueRef::Value(id) => Ok(DenseView::Borrowed(&values[id.0 as usize])),
    }
}

enum DenseView<'a> {
    Borrowed(&'a Dense),
}

impl<'a> DenseView<'a> {
    fn get(&self) -> &'a Dense {
        match self {
            DenseView::Borrowed(d) => d,
        }
    }
}

/// Evaluate every output of `program` by brute-force index loops.
pub fn eval_program(problem: &Problem, program: &Program, blocks: &Blocks) -> Result<Vec<Dense>> {
    let mut values: Vec<Dense> = Vec::with_capacity(program.values.len());
    for definition in &program.values {
        let value = match definition {
            ValueDef::Einsum(op) => {
                let code = &op.code;
                // Every index in the code iterates over its domain size.
                let indices: Vec<(u32, usize)> = code
                    .domains
                    .iter()
                    .map(|(label, domain)| (label.0, problem.sizes[domain] as usize))
                    .collect();
                let out_dims: Vec<usize> = code
                    .output
                    .iter()
                    .map(|l| problem.sizes[&code.domains[l]] as usize)
                    .collect();
                let mut out = Dense::zeros(out_dims);
                let mut assignment = vec![0usize; indices.len()];
                let coeff = coeff_f64(&op.coeff);
                // Resolve argument blocks once.
                let args: Vec<(DenseView, &Vec<truthmill::IndexId>)> = op
                    .args
                    .iter()
                    .zip(&code.inputs)
                    .map(|(a, labels)| block_of(blocks, *a, labels, code, &values).map(|b| (b, labels)))
                    .collect::<Result<_>>()?;
                let total: usize = indices.iter().map(|(_, s)| s).product();
                for _ in 0..total {
                    let mut value = coeff;
                    for (block, labels) in &args {
                        let block = block.get();
                        let index: Vec<usize> = labels
                            .iter()
                            .map(|l| {
                                let pos = indices.iter().position(|(id, _)| id == &l.0).unwrap();
                                assignment[pos]
                            })
                            .collect();
                        value *= block.data[block.offset(&index)];
                    }
                    let out_index: Vec<usize> = code
                        .output
                        .iter()
                        .map(|l| {
                            let pos = indices.iter().position(|(id, _)| id == &l.0).unwrap();
                            assignment[pos]
                        })
                        .collect();
                    let off = out.offset(&out_index);
                    out.data[off] += value;
                    // Increment mixed-radix counter.
                    for k in 0..indices.len() {
                        assignment[k] += 1;
                        if assignment[k] < indices[k].1 {
                            break;
                        }
                        assignment[k] = 0;
                    }
                }
                out
            }
            ValueDef::LinComb(op) => {
                let code = &op.code;
                let out_dims: Vec<usize> = code
                    .output
                    .iter()
                    .map(|l| problem.sizes[&code.domains[l]] as usize)
                    .collect();
                let mut out = Dense::zeros(out_dims.clone());
                let total: usize = out_dims.iter().product();
                let mut assignment = vec![0usize; out_dims.len()];
                for _ in 0..total {
                    let mut value = 0.0;
                    for (coeff, (arg, labels)) in
                        op.coeff.iter().zip(op.args.iter().zip(&code.inputs))
                    {
                        let block = block_of(blocks, *arg, labels, code, &values)?;
                        let block = block.get();
                        let index: Vec<usize> = labels
                            .iter()
                            .map(|l| {
                                let pos = code.output.iter().position(|o| o == l).unwrap();
                                assignment[pos]
                            })
                            .collect();
                        value += coeff_f64(coeff) * block.data[block.offset(&index)];
                    }
                    let off = out.offset(&assignment);
                    out.data[off] = value;
                    for k in 0..assignment.len() {
                        assignment[k] += 1;
                        if assignment[k] < out_dims[k] {
                            break;
                        }
                        assignment[k] = 0;
                    }
                }
                out
            }
        };
        values.push(value);
    }
    Ok(program
        .outputs
        .iter()
        .map(|o| values[o.0 as usize].clone())
        .collect())
}
