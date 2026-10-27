//! Layout-aware scheduling: choose a physical axis order for every input
//! block and every intermediate, so that contractions lower to GEMM-shaped
//! calls. Loop-invariant inputs pay the layout conversion once at setup
//! (untimed); intermediates inherit their producer's output order for free.
//!
//! Heuristic v0 (the initial HL target):
//!  - For a binary contraction C = A * B with K = contracted axes, the
//!    GEMM-ideal column-major layouts are A: [free..., K...], B: [K...,
//!    free...], C: [A-free..., B-free...] ("free fastest, M= A-free, N=
//!    B-free").
//!  - Each block's storage layout is the ideal layout of its most expensive
//!    use (weighted by union FLOPs of the using op).
//!  - Each intermediate's output order is its producing op's ideal C order.
//!
//! Correctness: block data is permuted at setup and every use's label list is
//! rewritten with the same permutation, so semantics are preserved by
//! construction; the numerical oracle gates the whole path, and
//! truthmill::verify gates the pre-layout program transforms.

use std::collections::BTreeMap;

use truthmill::{Code, Coefficient, DomainId, IndexId, Problem, Program, ValueDef, ValueRef};

use crate::dense::{permute, BlockKey, Blocks};

/// Physical storage order per block: storage axis k holds logical axis
/// `perm[k]`.
pub type Layouts = BTreeMap<BlockKey, Vec<usize>>;

/// Blocks after setup-time permutation, keyed by original pattern, with the
/// applied permutation attached.
pub type PermutedBlocks = std::collections::HashMap<BlockKey, (crate::dense::Dense, Vec<usize>)>;

struct Use {
    weight: f64, // log10 union flops of the using op
    ideal: Vec<usize>,
}

fn union_log10_flops(code: &Code, sizes: &BTreeMap<DomainId, u64>) -> f64 {
    let mut labels: Vec<IndexId> = code.output.clone();
    for input in &code.inputs {
        for l in input {
            if !labels.contains(l) {
                labels.push(*l);
            }
        }
    }
    labels
        .iter()
        .map(|l| (sizes[&code.domains[l]] as f64).log10())
        .sum()
}

/// Ideal storage order (as a permutation: storage axis k = logical axis
/// `ideal[k]`) for argument `arg_pos` of a binary einsum.
fn ideal_for_binary_arg(code: &Code, arg_pos: usize) -> Option<Vec<usize>> {
    if code.inputs.len() != 2 {
        return None;
    }
    let a = &code.inputs[0];
    let b = &code.inputs[1];
    let contracted: Vec<IndexId> = a.iter().filter(|l| b.contains(l)).cloned().collect();
    let mine = &code.inputs[arg_pos];
    let free: Vec<usize> = mine
        .iter()
        .enumerate()
        .filter(|(_, l)| !contracted.contains(l))
        .map(|(k, _)| k)
        .collect();
    let contr: Vec<usize> = mine
        .iter()
        .enumerate()
        .filter(|(_, l)| contracted.contains(l))
        .map(|(k, _)| k)
        .collect();
    // A operand: free first; B operand: contracted first (see module docs).
    let mut ideal = if arg_pos == 0 {
        free.into_iter().chain(contr).collect::<Vec<_>>()
    } else {
        contr.into_iter().chain(free).collect::<Vec<_>>()
    };
    // Identity check: [0,1,2,..] means keep.
    if ideal.iter().enumerate().all(|(k, &p)| p == k) {
        ideal = (0..mine.len()).collect();
    }
    Some(ideal)
}

/// Ideal output order for a binary einsum: [A-free..., B-free...] as a
/// permutation of the current output axes.
fn ideal_output_order(code: &Code) -> Option<Vec<usize>> {
    if code.inputs.len() != 2 {
        return None;
    }
    let a = &code.inputs[0];
    let b = &code.inputs[1];
    let contracted: Vec<IndexId> = a.iter().filter(|l| b.contains(l)).cloned().collect();
    let a_free: Vec<IndexId> = a.iter().filter(|l| !contracted.contains(l)).cloned().collect();
    let b_free: Vec<IndexId> = b.iter().filter(|l| !contracted.contains(l)).cloned().collect();
    let want: Vec<IndexId> = a_free.into_iter().chain(b_free).collect();
    if want.len() != code.output.len() {
        return None; // traces / diagonals: leave alone
    }
    // storage axis k should hold current output axis perm[k]
    let perm: Option<Vec<usize>> = want
        .iter()
        .map(|l| code.output.iter().position(|o| o == l))
        .collect();
    perm
}

/// Compute block layouts from the program: for every block use, record the
/// ideal layout weighted by the op's union FLOPs; take the argmax per block.
pub fn plan_layouts(problem: &Problem, program: &Program) -> Layouts {
    let mut uses: BTreeMap<BlockKey, Vec<Use>> = BTreeMap::new();
    for definition in &program.values {
        let (code, args) = match definition {
            ValueDef::Einsum(op) => (&op.code, &op.args),
            ValueDef::LinComb(op) => (&op.code, &op.args),
        };
        let weight = union_log10_flops(code, &problem.sizes);
        for (pos, (arg, labels)) in args.iter().zip(&code.inputs).enumerate() {
            if let ValueRef::Tensor(tensor) = arg {
                let pattern: Vec<u32> = labels.iter().map(|l| code.domains[l].0).collect();
                let ideal = ideal_for_binary_arg(code, pos)
                    .unwrap_or_else(|| (0..labels.len()).collect());
                uses.entry((tensor.0, pattern)).or_default().push(Use { weight, ideal });
            }
        }
    }
    let mut layouts = Layouts::new();
    if std::env::var("H_NO_BLOCK").is_ok() {
        return layouts;
    }
    for (key, uses) in uses {
        // Aggregate weight per distinct ideal layout.
        let mut by_layout: BTreeMap<Vec<usize>, f64> = BTreeMap::new();
        for u in uses {
            *by_layout.entry(u.ideal).or_default() += u.weight;
        }
        let best = by_layout
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(l, _)| l)
            .unwrap();
        if let Ok(only) = std::env::var("H_ONLY_BLOCK") {
            let want = format!("{}:{:?}", key.0, key.1);
            if want != only {
                continue;
            }
        }
        layouts.insert(key, best);
    }
    layouts
}

/// Apply layouts to input blocks (setup-time permutation; untimed).
///
/// Blocks stay keyed by their ORIGINAL domain pattern (permuted patterns of
/// distinct blocks can collide, e.g. g[v,o,o,v] vs g[v,o,v,o]); the
/// permutation travels with the block and is applied to the argument labels
/// at resolve time in the interpreter.
pub fn apply_layouts_to_blocks(blocks: &Blocks, layouts: &Layouts) -> PermutedBlocks {
    let mut out = PermutedBlocks::new();
    for (key, dense) in blocks {
        let perm = layouts.get(key).cloned().unwrap_or_else(|| (0..dense.dims.len()).collect());
        out.insert(key.clone(), (permute(dense, &perm, 1), perm));
    }
    out
}

/// Rewrite the program so block-arg label lists match the permuted storage,
/// and reorder intermediate output axes to GEMM-ideal order. Intermediate
/// output reordering is positional: every consumer's label list for that
/// value must be permuted by the same permutation (two-pass).
pub fn apply_layouts_to_program(program: &Program, layouts: &Layouts) -> Program {
    // Pass 1: desired output permutation per value (None = keep).
    let mut value_perms: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let no_inter = std::env::var("H_NO_INTER").is_ok();
    for (id, definition) in program.values.iter().enumerate() {
        if no_inter { break; }
        if let ValueDef::Einsum(op) = definition {
            if let Some(perm) = ideal_output_order(&op.code) {
                if !perm.iter().enumerate().all(|(k, &p)| k == p) {
                    value_perms.insert(id, perm);
                }
            }
        }
    }
    // Pass 2: rewrite.
    let mut values = Vec::with_capacity(program.values.len());
    for (id, definition) in program.values.iter().enumerate() {
        let _ = layouts; // block layouts are applied at resolve time, not here
    let mut rewrite_code = |code: &mut Code, args: &[ValueRef]| {
            for (arg, labels) in args.iter().zip(code.inputs.iter_mut()) {
                match arg {
                    // Block layouts are handled by the interpreter at resolve
                    // time (see apply_layouts_to_blocks); the program keeps
                    // logical label order for tensor args.
                    ValueRef::Tensor(_) => {}
                    ValueRef::Value(v) => {
                        if let Some(perm) = value_perms.get(&(v.0 as usize)) {
                            let old = labels.clone();
                            for (k, &p) in perm.iter().enumerate() {
                                labels[k] = old[p];
                            }
                        }
                    }
                }
            }
        };
        let rewritten = match definition {
            ValueDef::Einsum(op) => {
                let mut code = op.code.clone();
                rewrite_code(&mut code, &op.args);
                if let Some(perm) = value_perms.get(&id) {
                    let old = code.output.clone();
                    for (k, &p) in perm.iter().enumerate() {
                        code.output[k] = old[p];
                    }
                }
                ValueDef::Einsum(truthmill::Einsum {
                    coeff: op.coeff,
                    code,
                    args: op.args.clone(),
                })
            }
            ValueDef::LinComb(op) => {
                let mut code = op.code.clone();
                rewrite_code(&mut code, &op.args);
                ValueDef::LinComb(truthmill::LinComb {
                    coeff: op.coeff.clone(),
                    code,
                    args: op.args.clone(),
                })
            }
        };
        values.push(rewritten);
    }
    Program {
        values,
        outputs: program.outputs.clone(),
    }
}

/// Debug helper: dump the chosen layouts.
pub fn debug_dump(problem: &Problem, program: &Program) {
    let layouts = plan_layouts(problem, program);
    for (k, v) in &layouts {
        if v.iter().enumerate().any(|(i, &p)| i != p) {
            eprintln!("layout {:?} -> {:?}", k, v);
        }
    }
    let rewritten = apply_layouts_to_program(program, &layouts);
    for (id, def) in rewritten.values.iter().enumerate() {
        if let ValueDef::Einsum(op) = def {
            let orig = match &program.values[id] {
                ValueDef::Einsum(o) => o,
                _ => unreachable!(),
            };
            if op.code != orig.code {
                eprintln!("v{id} code changed: {:?} -> {:?}", orig.code.output, op.code.output);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use truthmill::{Einsum, IndexId, TensorId, ValueId};

    /// out[a,b] = sum_x T0[a,x] * T0[b,x]; T0 has block pattern [dom0, dom1].
    /// The two uses disagree on the ideal layout, forcing a nontrivial perm.
    fn toy_problem() -> (Problem, Program) {
        let mut sizes = BTreeMap::new();
        sizes.insert(DomainId(0), 2);
        sizes.insert(DomainId(1), 3);
        let program = Program {
            values: vec![ValueDef::Einsum(Einsum {
                coeff: Coefficient::from_integer(1.into()),
                code: Code {
                    inputs: vec![
                        vec![IndexId(0), IndexId(2)],
                        vec![IndexId(1), IndexId(2)],
                    ],
                    output: vec![IndexId(0), IndexId(1)],
                    domains: BTreeMap::from([
                        (IndexId(0), DomainId(0)),
                        (IndexId(1), DomainId(0)),
                        (IndexId(2), DomainId(1)),
                    ]),
                },
                args: vec![ValueRef::Tensor(TensorId(0)), ValueRef::Tensor(TensorId(0))],
            })],
            outputs: vec![ValueId(0)],
        };
        let problem = Problem {
            sizes,
            symmetries: BTreeMap::new(),
            reference: program.clone(),
        };
        (problem, program)
    }

    #[test]
    fn layout_transform_preserves_semantics() {
        let (problem, program) = toy_problem();
        let layouts = plan_layouts(&problem, &program);
        eprintln!("layouts: {layouts:?}");
        let blocks = crate::dense::generate_inputs(&problem, 7);
        let expected = crate::oracle::eval_program(&problem, &program, &blocks).unwrap();
        let rewritten = apply_layouts_to_program(&program, &layouts);
        let permuted = apply_layouts_to_blocks(&blocks, &layouts);
        let tensor_blocks = crate::interp::to_tensor_blocks_permuted(&permuted).unwrap();
        let mut backend = tenferro_cpu::CpuBackend::with_threads(1).unwrap();
        let actual = crate::interp::run_cold(&rewritten, &tensor_blocks, &mut backend).unwrap();
        let data = actual[0].as_slice::<f64>().unwrap();
        let mut max_diff = 0.0f64;
        for (x, y) in expected[0].data.iter().zip(data) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff < 1e-12, "max diff {max_diff}");
    }
}
