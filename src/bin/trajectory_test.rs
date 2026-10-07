// trajectory_test.rs — Temporal-gate validation.
// Question: does gating STAIR's candidate set by a time window help when
// content actually correlates with time? (On the word corpus it hurt:
// 66% -> 21% recall at +/-2 s, because words have no meaningful time.)
// Corpus: 50,000 synthetic embeddings on an Ornstein-Uhlenbeck trajectory
// in 768-d (mean-reverting random walk, autocorrelation length ~5 s),
// time = i * 0.1 s. Truth: brute-force top-10 by dot product.
// Config: STAIR L=4, radius=2, gate widths 0.25 / 0.5 / 1 / 2 / inf s.
// SYNTHETIC corpus: purpose is gate validation, NOT performance claims.
// Run: cargo run --release --bin trajectory_test

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::time::Instant;

const N: usize = 50_000;
const THETA: f32 = 0.02;      // mean-reversion rate (~50-step memory)
const DRIFT: f32 = 0.05;      // per-step innovation std
const QUERY_NOISE: f32 = 0.1; // per-dim query perturbation
const N_DIRS: usize = 32;
const L_TABLES: usize = 4;
const RADIUS: u32 = 2;
const K: usize = 10;
const N_QUERIES: usize = 40;
const SEED: u64 = 0x51DE_BEEF;

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
    let step = (embs.len() / 10_000).max(1);
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
    let mut rng = Rng::new(SEED);
    println!("Generating OU trajectory corpus: {} x 768 ...", N);
    let stat_std = DRIFT / (2.0 * THETA).sqrt();
    let mut embs: Vec<Vec<f32>> = Vec::with_capacity(N);
    let mut cur = vec![0.0f32; 768];
    for j in 0..768 { cur[j] = (rng.gauss() as f32) * stat_std; }
    for _ in 0..N {
        embs.push(cur.clone());
        for j in 0..768 { cur[j] += -THETA * cur[j] + (rng.gauss() as f32) * DRIFT; }
    }
    let times: Vec<f32> = (0..N).map(|i| i as f32 * 0.1).collect();

    let prefix_bits: u32 = ((N as f64).log2().ceil() as i64).clamp(4, 22) as u32;
    println!("Building {} tables (prefix={} bits)...", L_TABLES, prefix_bits);
    let t0 = Instant::now();
    let tables: Vec<Table> = (0..L_TABLES).map(|t| build_table(&embs, prefix_bits,
        SEED ^ (0x9E3779B97F4A7C15u64).wrapping_mul(t as u64 + 1), t)).collect();
    println!("Built in {:.1?}\n", t0.elapsed());
    let masks = probe_masks(prefix_bits, RADIUS);

    // queries + brute-force truth
    let mut queries = Vec::new();
    let mut truths = Vec::new();
    let mut srcs = Vec::new();
    let mut bf_time = 0.0f64;
    for _ in 0..N_QUERIES {
        let src = (rng.next_u64() as usize) % N;
        let q: Vec<f32> = embs[src].iter()
            .map(|a| a + (rng.gauss() as f32) * QUERY_NOISE).collect();
        let t1 = Instant::now();
        let mut h: BinaryHeap<Score> = BinaryHeap::new();
        for (i, e) in embs.iter().enumerate() {
            let mut s = 0.0f32;
            for j in 0..768 { s += q[j] * e[j]; }
            if h.len() < K { h.push(Score(s, i as u32)); }
            else if s > h.peek().unwrap().0 { h.pop(); h.push(Score(s, i as u32)); }
        }
        bf_time += t1.elapsed().as_secs_f64();
        truths.push(h.into_iter().map(|Score(_, i)| i).collect::<HashSet<u32>>());
        queries.push(q);
        srcs.push(src);
    }
    println!("Brute force: {:.1} ms/query\n", 1e3 * bf_time / N_QUERIES as f64);

    // how temporally spread is the truth? (interprets the gate results)
    let (mut sum_d, mut max_d) = (0.0f64, 0usize);
    for (qi, t) in truths.iter().enumerate() {
        for i in t {
            let d = (*i as isize - srcs[qi] as isize).unsigned_abs();
            sum_d += d as f64;
            max_d = max_d.max(d);
        }
    }
    let cnt = (N_QUERIES * K) as f64;
    println!("Truth temporal spread: mean |dt| = {:.1} steps ({:.2} s), max = {} steps ({:.2} s)",
             sum_d / cnt, 0.1 * sum_d / cnt, max_d, 0.1 * max_d as f64);
    println!("(On the word corpus the same +/-2 s gate cut recall 66% -> 21%.)\n");

    println!("{:>7} | {:>10} | {:>9} | {:>8}", "gate(s)", "recall@10", "us/q", "cand/q");
    println!("{}", "-".repeat(45));
    let gates = [0.25f32, 0.5, 1.0, 2.0, f32::INFINITY];
    for &g in &gates {
        let (mut rec, mut tt) = (0.0f64, 0.0f64);
        let mut cand_sum = 0usize;
        for (qi, q) in queries.iter().enumerate() {
            let t1 = Instant::now();
            let mut seen: HashSet<u32> = HashSet::new();
            for table in &tables {
                let code = sketch_of(&table.dirs, &table.mean, q);
                let prefix = code >> (64 - prefix_bits);
                for &m in &masks {
                    if let Some(idxs) = table.buckets.get(&(prefix ^ m)) {
                        for &i in idxs { seen.insert(i); }
                    }
                }
            }
            cand_sum += seen.len();
            let mut scored: Vec<(f32, u32)> = Vec::with_capacity(seen.len());
            for &i in &seen {
                let iu = i as usize;
                if (times[iu] - times[srcs[qi]]).abs() > g { continue; }
                let e = &embs[iu];
                let mut s = 0.0f32;
                for j in 0..768 { s += q[j] * e[j]; }
                scored.push((s, i));
            }
            scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0)
                                     .unwrap_or(std::cmp::Ordering::Equal));
            tt += t1.elapsed().as_secs_f64();
            let hits = scored.iter().take(K)
                .filter(|(_, i)| truths[qi].contains(i)).count();
            rec += hits as f64 / K as f64;
        }
        let qn = N_QUERIES as f64;
        let label = if g.is_infinite() { "inf".into() } else { format!("{:.2}", g) };
        println!("{:>7} | {:>9.1}% | {:>7.0} | {:>8.0}",
                 label, 100.0 * rec / qn, 1e6 * tt / qn, cand_sum as f64 / qn);
    }
    println!("\nExpected: recall roughly flat from inf down to ~1 s (truth is time-local),");
    println!("then dropping at 0.25-0.5 s; time falling monotonically with tighter gates.");
    println!("Together with the word-corpus result, this is the 'when gating helps' figure.");
}
