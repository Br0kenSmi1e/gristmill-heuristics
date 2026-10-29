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
    collect_patterns_in(&problem.reference)
}

/// Collect (tensor, domain pattern) pairs used by one program.
pub fn collect_patterns_in(
    program: &truthmill::Program,
) -> BTreeMap<TensorId, Vec<Vec<DomainId>>> {
    let mut patterns: BTreeMap<TensorId, Vec<Vec<DomainId>>> = BTreeMap::new();
    for definition in &program.values {
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

/// Generate random input blocks for a problem, one per (tensor, pattern).
/// Blocks related by a declared symmetry generator (possibly across domain
/// patterns) share one randomly generated representative: the representative
/// is projected onto its within-block stabilizer group, and related blocks are
/// derived by permute+sign. This enforces cross-block symmetry relations, so
/// candidates that exploit antisymmetry (e.g. gristmill's) check correctly.
pub fn generate_inputs(problem: &Problem, seed: u64) -> Blocks {
    generate_inputs_for(problem, &[], seed)
}

/// As `generate_inputs`, but also covering patterns used by extra programs
/// (e.g. external candidates whose accesses differ from the reference).
pub fn generate_inputs_for(
    problem: &Problem,
    extra_programs: &[&truthmill::Program],
    seed: u64,
) -> Blocks {
    let mut patterns = collect_patterns(problem);
    for program in extra_programs {
        for (tensor, pats) in collect_patterns_in(program) {
            let entry = patterns.entry(tensor).or_default();
            for pat in pats {
                if !entry.contains(&pat) {
                    entry.push(pat);
                }
            }
        }
    }
    let mut blocks = HashMap::new();
    for (tensor, pats) in &patterns {
        let generators = problem
            .symmetries
            .get(tensor)
            .cloned()
            .unwrap_or_default();
        // Breadth-first over patterns linked by generators.
        let mut filled: HashMap<Vec<u32>, ()> = HashMap::new();
        for representative in pats {
            let rep_key: Vec<u32> = representative.iter().map(|d| d.0).collect();
            if filled.contains_key(&rep_key) {
                continue;
            }
            // Generate the representative: random, projected on stabilizer.
            let dims: Vec<usize> = representative
                .iter()
                .map(|d| problem.sizes[d] as usize)
                .collect();
            let len: usize = dims.iter().product();
            let mut key = seed;
            for &dm in representative {
                key = key.wrapping_mul(1_000_003).wrapping_add(dm.0 as u64 + 7);
            }
            key = key.wrapping_mul(1_000_003).wrapping_add(tensor.0 as u64 + 13);
            let mut rng = ChaCha8Rng::seed_from_u64(key);
            let data: Vec<f64> = (0..len).map(|_| rng.random::<f64>() - 0.5).collect();
            let block = Dense { dims, data };
            let group = stabilizer_group(&generators, representative);
            let block = project(&block, &group);
            // BFS propagation along generators.
            let mut queue = vec![(rep_key.clone(), block)];
            while let Some((cur_key, cur_block)) = queue.pop() {
                if blocks.contains_key(&(tensor.0, cur_key.clone())) {
                    continue;
                }
                blocks.insert((tensor.0, cur_key.clone()), cur_block.clone());
                filled.insert(cur_key.clone(), ());
                for generator in &generators {
                    // q[k] = cur[p[k]]: storage axis k holds logical axis p[k].
                    let q: Vec<u32> = generator
                        .permutation
                        .iter()
                        .map(|&p| cur_key[p])
                        .collect();
                    // Propagate through the FULL pattern orbit, including
                    // unused intermediates. Canonicalization uses composed
                    // symmetries; filtering to used patterns can leave the
                    // endpoints of a composed relation independently random.
                    if filled.contains_key(&q) {
                        continue;
                    }
                    let image = permute(&cur_block, &generator.permutation, generator.sign);
                    queue.push((q, image));
                }
            }
        }
    }
    blocks
}

/// Total input-block bytes implied by the problem's access patterns,
/// BEFORE generating anything (early memory guard).
pub fn estimate_input_bytes(problem: &Problem) -> usize {
    collect_patterns(problem)
        .values()
        .flat_map(|pats| pats.iter())
        .map(|pat| pat.iter().map(|d| problem.sizes[d] as usize).product::<usize>() * 8)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_mp3_inputs_satisfy_declared_symmetries() {
        let mut problem = Problem::from_json(include_str!(
            "../../../benches/chemistry/mp3.json"
        ))
        .unwrap();
        problem.sizes.insert(DomainId(0), 3);
        problem.sizes.insert(DomainId(1), 5);
        let candidate: truthmill::Program = serde_json::from_str(include_str!(
            "../../../benches/candidates/mp3_selected.json"
        ))
        .unwrap();
        let blocks = generate_inputs_for(&problem, &[&candidate], 0);
        for ((tensor, pattern), block) in &blocks {
            let pattern: Vec<DomainId> = pattern.iter().map(|d| DomainId(*d)).collect();
            let group = stabilizer_group(&problem.symmetries[&TensorId(*tensor)], &pattern);
            for (permutation, sign) in group {
                let image = permute(block, &permutation, sign);
                let max_diff = block
                    .data
                    .iter()
                    .zip(&image.data)
                    .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
                assert!(
                    max_diff < 1e-12,
                    "tensor {tensor} pattern {pattern:?} violates {permutation:?}/{sign}: {max_diff}"
                );
            }
            let generators = &problem.symmetries[&TensorId(*tensor)];
            let mut group = vec![((0..pattern.len()).collect::<Vec<_>>(), 1i8)];
            let mut position = 0;
            while position < group.len() {
                let (permutation, sign) = group[position].clone();
                for generator in generators {
                    let element = (
                        compose(&permutation, &generator.permutation),
                        sign * generator.sign,
                    );
                    if !group.contains(&element) {
                        group.push(element);
                    }
                }
                position += 1;
            }
            for (permutation, sign) in group {
                let related_pattern: Vec<u32> =
                    permutation.iter().map(|&p| pattern[p].0).collect();
                if let Some(related) = blocks.get(&(*tensor, related_pattern)) {
                    let image = permute(block, &permutation, sign);
                    let max_diff = related
                        .data
                        .iter()
                        .zip(&image.data)
                        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
                    assert!(
                        max_diff < 1e-12,
                        "tensor {tensor} pattern {pattern:?} violates composed cross-block symmetry {permutation:?}/{sign}: {max_diff}"
                    );
                }
            }
        }
    }
}
