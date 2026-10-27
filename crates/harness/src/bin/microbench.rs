//! Microbenchmark: the dominant CCD contraction shape
//!   out[c,d,i,j] = sum_{a,b} t2[a,b,i,j] * g[c,d,a,b]
//! under three lowerings:
//!   1. einsum as-is
//!   2. einsum with g pre-permuted to (a,b,c,d) layout
//!   3. raw 2-D matmul on column-major views (the GEMM ideal)
//! across thread counts. Tests H2 (layout/lowering overhead).

use rand::SeedableRng;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use tenferro_cpu::CpuBackend;
use tenferro_einsum::TensorEinsumExt;
use tenferro_tensor::Tensor;

fn rand_tensor(dims: Vec<usize>, seed: u64) -> Tensor {
    let len: usize = dims.iter().product();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let data: Vec<f64> = (0..len).map(|_| rng.random::<f64>() - 0.5).collect();
    Tensor::from_vec_col_major(dims, data).unwrap()
}

fn permute4(x: &Tensor, perm: [usize; 4], backend: &mut CpuBackend) -> Tensor {
    let labels: Vec<Vec<u32>> = vec![(0..4).map(|i| i as u32).collect()];
    let out: Vec<u32> = perm.iter().map(|&p| p as u32).collect();
    let subs = tenferro_einsum::EinsumSubscripts::new(&[&labels[0]], &out);
    [x].as_slice().einsum_subscripts(&subs, backend).unwrap()
}

fn time_it<F: FnMut(&mut CpuBackend) -> Tensor>(mut f: F, backend: &mut CpuBackend, iters: usize) -> f64 {
    // warmup
    let r = f(backend);
    std::hint::black_box(&r);
    let mut times = Vec::new();
    for _ in 0..iters {
        let start = std::time::Instant::now();
        let r = f(backend);
        std::hint::black_box(&r);
        times.push(start.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2]
}

fn main() {
    let v: usize = std::env::args().nth(1).map(|s| s.parse().unwrap()).unwrap_or(64);
    let o: usize = std::env::args().nth(2).map(|s| s.parse().unwrap()).unwrap_or(16);
    let iters = 5;
    let flops = 2.0 * (v as f64).powi(4) * (o as f64).powi(2);

    let t2 = rand_tensor(vec![v, v, o, o], 1); // [a,b,i,j]
    let g = rand_tensor(vec![v, v, v, v], 2); // [c,d,a,b]
    let g_perm = rand_tensor(vec![v, v, v, v], 2); // same data, will be permuted at setup

    for threads in [1usize, 16] {
        let mut backend = CpuBackend::with_threads(threads).unwrap();

        // variant 1: einsum as-is "abij,cdab->cdij"
        let ms1 = time_it(
            |b| [&t2, &g].as_slice().einsum("abij,cdab->cdij", b).unwrap(),
            &mut backend,
            iters,
        );

        // variant 2: pre-permute g to [a,b,c,d], then einsum "abij,ab cd->cdij"
        let g_abcd = permute4(&g_perm, [2, 3, 0, 1], &mut backend);
        let ms2perm = time_it(|b| permute4(&g_perm, [2, 3, 0, 1], b), &mut backend, iters);
        let ms2 = time_it(
            |b| [&t2, &g_abcd].as_slice().einsum("abij,abcd->cdij", b).unwrap(),
            &mut backend,
            iters,
        );

        // variant 3: reshape to 2-D and matmul: C[(c,d),(i,j)] = G[(c,d),(a,b)] * T[(a,b),(i,j)]
        // g[c,d,a,b] col-major: (c,d) fastest => already (M,K) with M stride 1.
        // t2[a,b,i,j] col-major: (a,b) fastest => already (K,N) with K stride 1.
        let g2d = Tensor::from_vec_col_major(
            vec![v * v, v * v],
            g.as_slice::<f64>().unwrap().to_vec(),
        )
        .unwrap();
        let t22d = Tensor::from_vec_col_major(
            vec![v * v, o * o],
            t2.as_slice::<f64>().unwrap().to_vec(),
        )
        .unwrap();
        let ms3 = time_it(
            |b| {
                [&g2d, &t22d]
                    .as_slice()
                    .einsum("mk,kn->mn", b)
                    .unwrap()
            },
            &mut backend,
            iters,
        );

        let report = |name: &str, ms: f64| {
            println!(
                "v={v} o={o} threads={threads:2} {name:24} {ms:9.2} ms  {:6.1} GF/s",
                flops / ms / 1e6
            );
        };
        report("einsum as-is", ms1);
        report("einsum pre-permuted", ms2);
        report("  (permute alone)", ms2perm);
        report("2d matmul einsum", ms3);
    }
}
