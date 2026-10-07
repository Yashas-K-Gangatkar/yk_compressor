// stair_590k.rs — STAIR at corpus scale (yk_words.bin, 589,933 x 768).
// Measures recall@10 and per-query time for multi-table STAIR vs brute-force
// truth, and WRITES benchmark_queries.bin (perturbed queries + exact truth)
// so the Python FAISS comparison uses IDENTICAL queries and truth.
//
// Run: cargo run --release --bin stair_590k
// RAM: ~3 GB. Time: ~2-5 min total.

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::io::{Read, Write};
use std::time::Instant;

const N_DIRS: usize = 32;
const K: usize = 10;
const Q_PER_SIGMA: usize = 30;
const SIGMAS: [f32; 3] = [0.0, 0.25, 0.5];
const L_TABLES: usize = 4;
const SEED: u64 = 0xDEADBEEF;

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1) }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        self.0 = x; x
    }
    fn gauss(&mut self) -> f64 {
        let u1 = (self.next_u64() >> 40) as f64 / (1u64 << 24) as f64 + 1e-12;
        let u2 = (self.next_u64() >> 40) as f64 / (1u64 << 24) as f64;
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

fn normalize(v: &mut [f32]) {
    let n: f32 = v.iter().map(|a| a * a).sum::<f32>().sqrt();
    if n > 1e-12 { for a in v.iter_mut() { *a /= n; } }
}

fn learn_directions(sample: &[Vec<f32>], n_dirs: usize, rng: &mut Rng,
                    iters: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    let d = sample[0].len();
    let mut mean = vec![0.0f32; d];
    for x in sample { for j in 0..d { mean[j] += x[j]; } }
    let n = sample.len() as f32;
    for m in mean.iter_mut() { *m /= n; }
    let mut resid: Vec<Vec<f32>> = sample.iter()
        .map(|x| x.iter().zip(&mean).map(|(a, b)| a - b).collect()).collect();
    let mut dirs = Vec::with_capacity(n_dirs);
    for _ in 0..n_dirs {
        let mut v: Vec<f32> = (0..d).map(|_| rng.gauss() as f32).collect();
        normalize(&mut v);
        for _ in 0..iters {
            let mut next = vec![0.0f32; d];
            for x in &resid {
                let mut s = 0.0f32;
                for j in 0..d { s += x[j] * v[j]; }
                if s != 0.0 { for j in 0..d { next[j] += s * x[j]; } }
            }
            normalize(&mut next);
            v = next;
        }
        for x in resid.iter_mut() {
            let mut s = 0.0f32;
            for j in 0..d { s += x[j] * v[j]; }
            for j in 0..d { x[j] -= s * v[j]; }
        }
        dirs.push(v);
    }
    (dirs, mean)
}

fn sketch_of(dirs: &[Vec<f32>], mean: &[f32], emb: &[f32]) -> u64 {
    let mut code = 0u64;
    for (i, dir) in dirs.iter().enumerate() {
        let mut s = 0.0f32;
        for j in 0..emb.len() { s += (emb[j] - mean[j]) * dir[j]; }
        if s > 0.0 { code |= 1u64 << (63 - i); }
    }
    code
}

struct Table { dirs: Vec<Vec<f32>>, mean: Vec<f32>, buckets: HashMap<u64, Vec<u32>> }

fn build_table(embs: &[Vec<f32>], prefix_bits: u32, seed: u64, tag: usize) -> Table {
    let t0 = Instant::now();
    let step = (embs.len() / 50_000).max(1);
    let sample: Vec<Vec<f32>> = (0..embs.len()).step_by(step)
        .map(|i| embs[i].clone()).collect();
    let mut rng = Rng::new(seed);
    let (dirs, mean) = learn_directions(&sample, N_DIRS, &mut rng, 5);
    drop(sample);
    let mut buckets: HashMap<u64, Vec<u32>> = HashMap::new();
    for (i, e) in embs.iter().enumerate() {
        let code = sketch_of(&dirs, &mean, e);
        buckets.entry(code >> (64 - prefix_bits)).or_default().push(i as u32);
    }
    println!("  table {}: {} buckets, {:?}", tag, buckets.len(), t0.elapsed());
    Table { dirs, mean, buckets }
}

fn probe_masks(p: u32, r: u32) -> Vec<u64> {
    let mut masks = vec![0u64];
    for kk in 1..=r {
        for m in 0..(1u64 << p) {
            if m.count_ones() == kk { masks.push(m); }
        }
    }
    masks
}

struct Score(f32, u32);
impl PartialEq for Score { fn eq(&self, o: &Self) -> bool { self.0 == o.0 && self.1 == o.1 } }
impl Eq for Score {}
impl PartialOrd for Score { fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) } }
impl Ord for Score { fn cmp(&self, o: &Self) -> std::cmp::Ordering { o.0.total_cmp(&self.0).then(o.1.cmp(&self.1)) } }

fn main() {
    let mut data = Vec::new();
    File::open("yk_words.bin").expect("yk_words.bin").read_to_end(&mut data).unwrap();
    let n = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    println!("Loaded {} embeddings ({} MB)", n, data.len() / 1024 / 1024);
    let mut embs: Vec<Vec<f32>> = Vec::with_capacity(n);
    let mut pos = 4;
    for _ in 0..n {
        pos += 4; // timecode (unused: words have no meaningful time)
        let mut e = vec![0.0f32; 768];
        for j in 0..768 {
            let off = pos + j * 4;
            e[j] = f32::from_le_bytes(data[off..off + 4].try_into().unwrap());
        }
        pos += 3072;
        embs.push(e);
    }
    drop(data);

    let prefix_bits: u32 = ((n as f64).log2().ceil() as i64).clamp(4, 22) as u32;
    println!("prefix={} bits, building {} tables...", prefix_bits, L_TABLES);
    let t0 = Instant::now();
    let tables: Vec<Table> = (0..L_TABLES).map(|t| build_table(
        &embs, prefix_bits,
        SEED ^ (0x9E3779B97F4A7C15u64).wrapping_mul(t as u64 + 1), t)).collect();
    println!("All tables built in {:.1?}\n", t0.elapsed());

    let masks_r1 = probe_masks(prefix_bits, 1);
    let masks_r2 = probe_masks(prefix_bits, 2);

    let mut rng = Rng::new(SEED);

    // per-dimension std for noise scaling
    let mut sum = vec![0.0f32; 768];
    let mut sumsq = vec![0.0f32; 768];
    for e in &embs { for j in 0..768 { sum[j] += e[j]; sumsq[j] += e[j] * e[j]; } }
    let dim_std: Vec<f32> = (0..768).map(|j| {
        let m = sum[j] / n as f32;
        ((sumsq[j] / n as f32) - m * m).max(0.0).sqrt()
    }).collect();

    // generate queries + brute-force truth (timed = honest brute-force baseline)
    let mut queries: Vec<Vec<f32>> = Vec::new();
    let mut truths: Vec<Vec<u32>> = Vec::new();
    let mut truth_time = 0.0f64;
    println!("Computing brute-force truth for {} queries...", SIGMAS.len() * Q_PER_SIGMA);
    for &sigma in &SIGMAS {
        for _ in 0..Q_PER_SIGMA {
            let src = (rng.next_u64() as usize) % n;
            let q: Vec<f32> = embs[src].iter().enumerate()
                .map(|(j, a)| a + (rng.gauss() as f32) * sigma * dim_std[j]).collect();
            let t1 = Instant::now();
            let mut h: BinaryHeap<Score> = BinaryHeap::new();
            for (i, e) in embs.iter().enumerate() {
                let mut s = 0.0f32;
                for j in 0..768 { s += q[j] * e[j]; }
                if h.len() < K { h.push(Score(s, i as u32)); }
                else if s > h.peek().unwrap().0 { h.pop(); h.push(Score(s, i as u32)); }
            }
            truth_time += t1.elapsed().as_secs_f64();
            truths.push(h.into_iter().map(|Score(_, i)| i).collect());
            queries.push(q);
        }
    }
    let nq = queries.len();
    println!("Brute force: {:.1} ms/query (this is the O(N) baseline)\n",
             1e3 * truth_time / nq as f64);

    // write shared query file for the Python FAISS comparison
    let mut out = Vec::with_capacity(8 + nq * (768 + K) * 4);
    out.extend_from_slice(&(nq as u32).to_le_bytes());
    out.extend_from_slice(&(K as u32).to_le_bytes());
    for (qi, q) in queries.iter().enumerate() {
        for v in q { out.extend_from_slice(&v.to_le_bytes()); }
        for id in &truths[qi] { out.extend_from_slice(&id.to_le_bytes()); }
    }
    File::create("benchmark_queries.bin").unwrap().write_all(&out).unwrap();
    println!("Wrote benchmark_queries.bin ({} queries + truth)\n", nq);

    println!("{:>5} | {:>6} | {:>10} | {:>11} | {:>7}", "sigma", "L", "radius",
             "recall@10", "us/q");
    println!("{}", "-".repeat(50));
    for (si, &sigma) in SIGMAS.iter().enumerate() {
        let qs = &queries[si * Q_PER_SIGMA..(si + 1) * Q_PER_SIGMA];
        let ts = &truths[si * Q_PER_SIGMA..(si + 1) * Q_PER_SIGMA];
        for (radius, masks) in [(1u32, &masks_r1), (2u32, &masks_r2)] {
            for &l_use in &[1usize, 2, 4] {
                let mut rec = 0.0f64;
                let mut total_time = 0.0f64;
                let mut cand_sum = 0usize;
                for (qi, q) in qs.iter().enumerate() {
                    let t1 = Instant::now();
                    let mut seen: HashSet<u32> = HashSet::new();
                    for table in &tables[..l_use] {
                        let code = sketch_of(&table.dirs, &table.mean, q);
                        let prefix = code >> (64 - prefix_bits);
                        for &m in masks.iter() {
                            if let Some(idxs) = table.buckets.get(&(prefix ^ m)) {
                                for &i in idxs { seen.insert(i); }
                            }
                        }
                    }
                    cand_sum += seen.len();
                    let mut scored: Vec<(f32, u32)> = Vec::with_capacity(seen.len());
                    for &i in &seen {
                        let e = &embs[i as usize];
                        let mut s = 0.0f32;
                        for j in 0..768 { s += q[j] * e[j]; }
                        scored.push((s, i));
                    }
                    scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0)
                                             .unwrap_or(std::cmp::Ordering::Equal));
                    total_time += t1.elapsed().as_secs_f64();
                    let truth_set: HashSet<u32> = ts[qi].iter().copied().collect();
                    let hits = scored.iter().take(K)
                        .filter(|(_, i)| truth_set.contains(i)).count();
                    rec += hits as f64 / K as f64;
                }
                let qn = Q_PER_SIGMA as f64;
                println!("{:>5.2} | {:>6} | {:>6} | {:>9.1}% | {:>9.0}",
                         sigma, l_use, radius, 100.0 * rec / qn,
                         1e6 * total_time / qn);
            }
        }
        println!("{}", "-".repeat(50));
    }
    println!("\nHonest speedup = brute-force ms/query (printed above) / STAIR us/query.");
    println!("Truth was computed once up-front; STAIR timing excludes it.");
}
