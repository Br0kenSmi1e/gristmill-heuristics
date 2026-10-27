//! Dense column-major tensor blocks and signed permutation-symmetry handling.

use std::collections::{BTreeMap, HashMap};

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use truthmill::{DomainId, Problem, SymmetryGenerator, TensorId, ValueDef, ValueRef};

/// A dense column-major block of f64.
#[derive(Clone, Debug)]
pub struct Dense {
    pub dims: Vec<usize>,
    pub data: Vec<f64>,
}

impl Dense {
    pub fn zeros(dims: Vec<usize>) -> Self {
        let len = dims.iter().product();
        Dense {
            dims,
            data: vec![0.0; len],
        }
    }

    /// Column-major offset of a multi-index.
    pub fn offset(&self, index: &[usize]) -> usize {
        let mut offset = 0;
        let mut stride = 1;
        for (axis, &i) in index.iter().enumerate() {
            offset += i * stride;
            stride *= self.dims[axis];
        }
        offset
    }
}

/// Key for an input block: tensor id plus the domain pattern of one access.
pub type BlockKey = (u32, Vec<u32>);
pub type Blocks = HashMap<BlockKey, Dense>;

/// Collect every (tensor, domain pattern) pair used by the reference program.
pub fn collect_patterns(problem: &Problem) -> BTreeMap<TensorId, Vec<Vec<DomainId>>> {
    let mut patterns: BTreeMap<TensorId, Vec<Vec<DomainId>>> = BTreeMap::new();
    for definition in &problem.reference.values {
        let (code, args) = match definition {
            ValueDef::Einsum(op) => (&op.code, &op.args),
            ValueDef::LinComb(op) => (&op.code, &op.args),
        };
        for (arg, labels) in args.iter().zip(&code.inputs) {
            if let ValueRef::Tensor(tensor) = arg {
                let pattern: Vec<DomainId> = labels
                    .iter()
                    .map(|label| code.domains[label])
                    .collect();
                let entry = patterns.entry(*tensor).or_default();
                if !entry.contains(&pattern) {
                    entry.push(pattern);
                }
            }
        }
    }
    patterns
}

/// Compose two axis permutations: `compose(p, q)[k] = p[q[k]]`.
fn compose(p: &[usize], q: &[usize]) -> Vec<usize> {
    q.iter().map(|&k| p[k]).collect()
}

/// Apply a signed axis permutation: output axis `k` is input axis `perm[k]`,
/// with the given overall sign.
pub fn permute(block: &Dense, perm: &[usize], sign: i8) -> Dense {
    let rank = block.dims.len();
    let mut inverse = vec![0usize; rank];
    for (k, &p) in perm.iter().enumerate() {
        inverse[p] = k;
    }
    let new_dims: Vec<usize> = perm.iter().map(|&p| block.dims[p]).collect();
    let mut out = Dense::zeros(new_dims);
    let mut index = vec![0usize; rank];
    let total: usize = block.data.len();
    for linear in 0..total {
        // Decompose `linear` into the output multi-index (column-major).
        let mut rest = linear;
        for axis in 0..rank {
            index[axis] = rest % out.dims[axis];
            rest /= out.dims[axis];
        }
        let mut old_index = vec![0usize; rank];
        for axis in 0..rank {
            old_index[axis] = index[inverse[axis]];
        }
        out.data[linear] = (sign as f64) * block.data[block.offset(&old_index)];
    }
    out
}

/// The subgroup of `generators` whose permutation preserves `pattern`
/// elementwise, as a closure under composition. Includes the identity.
pub fn stabilizer_group(
    generators: &[SymmetryGenerator],
    pattern: &[DomainId],
) -> Vec<(Vec<usize>, i8)> {
    let rank = pattern.len();
    let identity: Vec<usize> = (0..rank).collect();
    let mut group: Vec<(Vec<usize>, i8)> = vec![(identity, 1)];
    let mut queue: Vec<(Vec<usize>, i8)> = generators
        .iter()
        .filter(|g| {
            g.permutation
                .iter()
                .enumerate()
                .all(|(k, &p)| pattern[p] == pattern[k])
        })
        .map(|g| (g.permutation.clone(), g.sign))
        .collect();
    // Breadth-first closure under composition; every newly discovered element
    // is requeued so it is composed with the whole group found so far.
    while let Some((perm, sign)) = queue.pop() {
        let snapshot = group.clone();
        let mut candidates: Vec<(Vec<usize>, i8)> = vec![(compose(&perm, &perm), sign * sign)];
        for (gperm, gsign) in &snapshot {
            candidates.push((compose(&perm, gperm), sign * gsign));
            candidates.push((compose(gperm, &perm), gsign * sign));
        }
        for candidate in candidates {
            if !group.iter().any(|(p, _)| *p == candidate.0) {
                group.push(candidate.clone());
                queue.push(candidate);
            }
        }
    }
    group
}

/// Project a block onto its signed symmetry subgroup (averaging over the
/// stabilizer group of its domain pattern).
pub fn project(block: &Dense, group: &[(Vec<usize>, i8)]) -> Dense {
    if group.len() <= 1 {
        return block.clone();
    }
    let mut acc = Dense::zeros(block.dims.clone());
    for (perm, sign) in group {
        let image = permute(block, perm, *sign);
        for (a, v) in acc.data.iter_mut().zip(image.data) {
            *a += v;
        }
    }
    let scale = 1.0 / group.len() as f64;
    for a in acc.data.iter_mut() {
        *a *= scale;
    }
    acc
}

/// Generate random input blocks for a problem, one per (tensor, pattern),
/// projected onto the within-block symmetry group.
pub fn generate_inputs(problem: &Problem, seed: u64) -> Blocks {
    let patterns = collect_patterns(problem);
    let mut blocks = HashMap::new();
    for (tensor, pats) in &patterns {
        let generators = problem
            .symmetries
            .get(tensor)
            .cloned()
            .unwrap_or_default();
        for pattern in pats {
            let dims: Vec<usize> = pattern
                .iter()
                .map(|d| problem.sizes[d] as usize)
                .collect();
            let len: usize = dims.iter().product();
            // Deterministic per-block seed.
            let mut key = seed;
            for &d in pattern {
                key = key.wrapping_mul(1_000_003).wrapping_add(d.0 as u64 + 7);
            }
            key = key.wrapping_mul(1_000_003).wrapping_add(tensor.0 as u64 + 13);
            let mut rng = ChaCha8Rng::seed_from_u64(key);
            let data: Vec<f64> = (0..len).map(|_| rng.random::<f64>() - 0.5).collect();
            let block = Dense { dims, data };
            let group = stabilizer_group(&generators, pattern);
            let block = project(&block, &group);
            blocks.insert((tensor.0, pattern.iter().map(|d| d.0).collect()), block);
        }
    }
    blocks
}
