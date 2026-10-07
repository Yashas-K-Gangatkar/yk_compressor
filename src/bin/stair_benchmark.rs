// stair_benchmark.rs  (v2 — bug-fix + honesty pass)
//
// What this benchmark IS:
//   - Perturbed-query recall evaluation against brute-force dot-product truth.
//   - Compares: (a) the old exact-match YK dict, (b) STAIR (sketch+multi-probe+rerank).
//
// What this benchmark is NOT:
//   - NOT a comparison against transformer attention. Attention aggregates over
//     tokens whose identity is unknown; every index here requires knowing the
//     query key or query vector in advance.
//
// Input: real_200.bin format -> u32 count, then per item: f32 timecode, 768 x f32.
// Run:   cargo run --release --bin stair_benchmark -- real_200.bin

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::io::Read;
use std::time::Instant;

// ---------------- config (tune these, one place) ----------------
const N_DIRS: usize = 32;      // sketch directions
const PROBE_RADIUS: u32 = 2;   // Hamming radius around query prefix
const K: usize = 10;           // recall@K
const GATE_S: f32 = 2.0;       // temporal gate half-width (seconds)
const Q_PER_SIGMA: usize = 50; // queries per noise level
const SEED: u64 = 0xDEADBEEF;

// ---------------- deterministic PRNG ----------------
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1) }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        self.0 = x; x
    }
    fn uniform(&mut self) -> f64 { (self.next_u64() >> 40) as f64 / (1u64 << 24) as f64 }
    fn gauss(&mut self) -> f64 {
        let u1 = (self.next_u64() >> 40) as f64 / (1u64 << 24) as f64 + 1e-12;
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

fn normalize(v: &mut [f32]) {
    let n: f32 = v.iter().map(|a| a * a).sum::<f32>().sqrt();
    if n > 1e-12 { for a in v.iter_mut() { *a /= n; } }
}

// -------- data-dependent directions (power iteration + deflation) --------
fn learn_directions(sample: &[Vec<f32>], n_dirs: usize, rng: &mut Rng,
                    iters: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    let d = sample[0].len();
    let mut mean = vec![0.0f32; d];
    for x in sample { for j in 0..d { mean[j] += x[j]; } }
    let n = sample.len() as f32;
    for m in mean.iter_mut() { *m /= n; }

    let mut resid: Vec<Vec<f32>> = sample.iter()
        .map(|x| x.iter().zip(&mean).map(|(a, b)| a - b).collect())
        .collect();

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

// -------- top-k without full sort (heap, survives 590K) --------
struct Score(f32, u32);
impl PartialEq for Score { fn eq(&self, o: &Self) -> bool { self.0 == o.0 && self.1 == o.1 } }
impl Eq for Score {}
impl PartialOrd for Score { fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) } }
impl Ord for Score { fn cmp(&self, o: &Self) -> std::cmp::Ordering { o.0.total_cmp(&self.0).then(o.1.cmp(&self.1)) } }

fn top_k(scores: impl Iterator<Item = (f32, u32)>, k: usize) -> Vec<(f32, u32)> {
    let mut h: BinaryHeap<Score> = BinaryHeap::new();
    for (s, i) in scores {
        if h.len() < k { h.push(Score(s, i)); }
        else if s > h.peek().unwrap().0 { h.pop(); h.push(Score(s, i)); }
    }
    let mut v: Vec<(f32, u32)> = h.into_iter().map(|Score(s, i)| (s, i)).collect();
    v.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    v
}

// ---------------- STAIR index ----------------
struct Stair {
    dirs: Vec<Vec<f32>>,
    mean: Vec<f32>,
    prefix_bits: u32,
    buckets: HashMap<u64, Vec<u32>>,
    masks: Vec<u64>,          // probes generated HERE — can never mismatch prefix
    times: Vec<f32>,
    modalities: Vec<u8>,
    embs: Vec<Vec<f32>>,
}

impl Stair {
    fn sketch(&self, emb: &[f32]) -> u64 {
        let mut code = 0u64;
        for (i, dir) in self.dirs.iter().enumerate() {
            let mut s = 0.0f32;
            for j in 0..emb.len() { s += (emb[j] - self.mean[j]) * dir[j]; }
            if s > 0.0 { code |= 1u64 << (63 - i); }
        }
        code
    }

    fn build(embs: Vec<Vec<f32>>, times: Vec<f32>, modalities: Vec<u8>,
             rng: &mut Rng) -> Stair {
        let t0 = Instant::now();
        let n = embs.len();
        // Adaptive prefix ~1 item/bucket. (Fixed 12 bits starved candidates at n=200.)
        let prefix_bits: u32 = ((n as f64).log2().ceil() as i64).clamp(4, 22) as u32;

        let sample_n = n.min(50_000);
        let step = (n / sample_n.max(1)).max(1);
        let sample: Vec<Vec<f32>> = (0..n).step_by(step).map(|i| embs[i].clone()).collect();
        let (dirs, mean) = learn_directions(&sample, N_DIRS, rng, 5);

        let mut s = Stair { dirs, mean, prefix_bits, buckets: HashMap::new(),
                            masks: Vec::new(), times, modalities, embs };
        for i in 0..n {
            let code = s.sketch(&s.embs[i]);
            s.buckets.entry(code >> (64 - s.prefix_bits)).or_default().push(i as u32);
        }
        s.masks = probe_masks(s.prefix_bits, PROBE_RADIUS);

        let maxb = s.buckets.values().map(|v| v.len()).max().unwrap_or(0);
        let avg = n as f64 / s.buckets.len().max(1) as f64;
        println!("STAIR build: {} items, prefix={} bits, {} buckets, {} probes/bucket, {:.2?}",
                 n, s.prefix_bits, s.buckets.len(), s.masks.len(), t0.elapsed());
        println!("Bucket stats: mean size {:.1}, max size {}", avg, maxb);
        s
    }

    // Returns (top-k ids, raw candidate count before filters) — the count is
    // the starvation diagnostic.
    fn query(&self, q: &[f32], modalities: &[u8], t_center: f32, window: f32,
             k: usize) -> (Vec<u32>, usize) {
        let qcode = self.sketch(q);
        let qprefix = qcode >> (64 - self.prefix_bits);
        let mut seen: HashSet<u32> = HashSet::new();
        for &m in &self.masks {
            if let Some(idxs) = self.buckets.get(&(qprefix ^ m)) {
                for &i in idxs { seen.insert(i); }
            }
        }
        let cand = seen.len();
        let mut scored: Vec<(f32, u32)> = Vec::with_capacity(cand);
        for &i in &seen {
            let iu = i as usize;
            if (self.times[iu] - t_center).abs() > window { continue; }
            if !modalities.contains(&self.modalities[iu]) { continue; }
            let e = &self.embs[iu];
            let mut s = 0.0f32;
            for j in 0..q.len() { s += q[j] * e[j]; }
            scored.push((s, i));
        }
        scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0)
                                 .unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        (scored.into_iter().map(|(_, i)| i).collect(), cand)
    }
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

// ---------------- old YK exact-match hash, unchanged ----------------
fn hash_yk(emb: &[f32]) -> u32 {
    let mut hash: u32 = 0;
    for i in 0..8.min(emb.len()) {
        let scaled = (emb[i] * 10000.0) as i32;
        hash = hash.wrapping_mul(31).wrapping_add(scaled as u32);
    }
    hash
}

// ---------------- benchmark ----------------
fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "real_200.bin".into());
    let mut data = Vec::new();
    File::open(&path).expect("open input bin").read_to_end(&mut data).unwrap();

    let mut pos = 0;
    let n = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    pos += 4;
    let mut embs = Vec::with_capacity(n);
    let mut times = Vec::with_capacity(n);
    for _ in 0..n {
        let t = f32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let mut e = vec![0.0f32; 768];
        for j in 0..768 {
            let off = pos + j * 4;
            e[j] = f32::from_le_bytes(data[off..off + 4].try_into().unwrap());
        }
        pos += 3072;
        times.push(t);
        embs.push(e);
    }
    let mods = vec![1u8; n];
    println!("Loaded {} embeddings from {}", n, path);
    println!("Config: dirs={} prefix=adaptive radius={} K={} gate=±{}s queries/sigma={} seed={}\n",
             N_DIRS, PROBE_RADIUS, K, GATE_S, Q_PER_SIGMA, SEED);

    let mut rng = Rng::new(SEED);
    let stair = Stair::build(embs.clone(), times.clone(), mods.clone(), &mut rng);

    // old exact-match system (single-value dict)
    let mut old: HashMap<(u32, u32, u32), u32> = HashMap::new();
    for i in 0..n {
        old.insert((hash_yk(&embs[i]), 1u32, (times[i] * 10.0) as u32), i as u32);
    }

    // per-dimension std -> sigma is interpretable per dimension
    let mut sum = vec![0.0f32; 768];
    let mut sumsq = vec![0.0f32; 768];
    for e in &embs { for j in 0..768 { sum[j] += e[j]; sumsq[j] += e[j] * e[j]; } }
    let dim_std: Vec<f32> = (0..768).map(|j| {
        let m = sum[j] / n as f32;
        ((sumsq[j] / n as f32) - m * m).max(0.0).sqrt()
    }).collect();

    let sigmas: [f32; 5] = [0.0, 0.1, 0.25, 0.5, 1.0];

    println!("{:>5} | {:>9} | {:>11} | {:>15} | {:>7} | {:>9}",
             "sigma", "old hit@1", "STAIR r@10", "STAIR+gate r@10", "cand/q", "gate µs/q");
    println!("{}", "-".repeat(70));

    for &sigma in &sigmas {
        let (mut hit_old, mut rec_stair, mut rec_gate) = (0.0f64, 0.0f64, 0.0f64);
        let mut gate_time = 0.0f64;
        let mut cand_sum = 0usize;

        for _qi in 0..Q_PER_SIGMA {
            let src = (rng.next_u64() as usize) % n;
            let q: Vec<f32> = embs[src].iter().enumerate()
                .map(|(j, a)| a + (rng.gauss() as f32) * sigma * dim_std[j])
                .collect();
            let t_q = times[src];

            // ground truth: exact top-K by dot product (heap, no full sort)
            let truth = top_k(embs.iter().enumerate().map(|(i, e)| {
                let mut s = 0.0f32;
                for j in 0..768 { s += q[j] * e[j]; }
                (s, i as u32)
            }), K);
            let truth_set: HashSet<u32> = truth.iter().map(|(_, i)| *i).collect();

            // old system (favorable: its time bucket is NOT perturbed)
            let hit = old.get(&(hash_yk(&q), 1u32, (t_q * 10.0) as u32))
                .map(|i| truth_set.contains(i)).unwrap_or(false);
            hit_old += hit as usize as f64;

            let (r, c) = stair.query(&q, &[1], t_q, f32::INFINITY, K);
            cand_sum += c;
            rec_stair += r.iter().filter(|i| truth_set.contains(i)).count() as f64 / K as f64;

            let t0 = Instant::now();
            let (rg, _) = stair.query(&q, &[1], t_q, GATE_S, K);
            gate_time += t0.elapsed().as_secs_f64();
            rec_gate += rg.iter().filter(|i| truth_set.contains(i)).count() as f64 / K as f64;
        }
        let qn = Q_PER_SIGMA as f64;
        println!("{:>5.2} | {:>8.0}% | {:>10.1}% | {:>14.1}% | {:>7.1} | {:>9.2}",
                 sigma, 100.0 * hit_old / qn, 100.0 * rec_stair / qn,
                 100.0 * rec_gate / qn, cand_sum as f64 / qn, 1e6 * gate_time / qn);
    }

    println!("\nSemantics:");
    println!("- old hit@1: the old system returns AT MOST ONE embedding (the dict hit).");
    println!("  100% = its single answer was a true top-{} neighbor. It returns", K);
    println!("  NOTHING at sigma>0 — it is an exact-match dictionary, not a similarity index.");
    println!("- STAIR r@10 / gate r@10: standard recall@{} vs brute-force truth.", K);
    println!("- cand/q: candidates entering rerank. If this is < ~3x {}, recall is", K);
    println!("  candidate-starved -> raise PROBE_RADIUS or lower prefix bits.");
    println!("- The old system's time bucket was NOT perturbed; this favors it.");
}
