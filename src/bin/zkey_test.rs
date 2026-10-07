// zkey_test.rs — Time as a KEY dimension, not a post-filter.
// Same OU trajectory corpus as trajectory_test (truth: mean |dt|=0.25s, max 0.6s).
// Modes:
//   A) ungated STAIR          (no time use)
//   B) post-filter gate       (current design: probe all, filter rerank)
//   C) Z-in-key gate          (bucket key = time_block + sketch prefix;
//                              probes only blocks in window -> 3D key)
//   D) time-range query       (Z-axis as PRIMARY key: rank all items in the
//                              window's blocks by dot product — the video
//                              use case, impossible in the old design)
// Run: cargo run --release --bin zkey_test
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::time::Instant;

const N: usize = 50_000;
const THETA: f32 = 0.02;
const DRIFT: f32 = 0.05;
const QUERY_NOISE: f32 = 0.1;
const N_DIRS: usize = 32;
const L_TABLES: usize = 4;
const RADIUS: u32 = 2;
const K: usize = 10;
const N_QUERIES: usize = 40;
const GATE: f32 = 0.6;          // covers measured truth spread
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
struct Table {
    dirs: Vec<Vec<f32>>, mean: Vec<f32>,
    // 2D key: (time_block << 32) | sketch_prefix  — the 3D grid, flattened
    zkey_buckets: HashMap<u64, Vec<u32>>,
    // 1D key: time_block only — for time-range queries (mode D)
    z_buckets: HashMap<u64, Vec<u32>>,
}
fn build_table(embs: &[Vec<f32>], times: &[f32], prefix_bits: u32,
               block_s: f32, seed: u64, tag: usize) -> Table {
    let step = (embs.len() / 10_000).max(1);
    let sample: Vec<Vec<f32>> = (0..embs.len()).step_by(step)
        .map(|i| embs[i].clone()).collect();
    let mut rng = Rng::new(seed);
    let (dirs, mean) = learn_directions(&sample, N_DIRS, &mut rng, 5);
    drop(sample);
    let mut zk: HashMap<u64, Vec<u32>> = HashMap::new();
    let mut z: HashMap<u64, Vec<u32>> = HashMap::new();
    for (i, e) in embs.iter().enumerate() {
        let prefix = sketch_of(&dirs, &mean, e) >> (64 - prefix_bits);
        let blk = (times[i] / block_s).floor() as u64;
        zk.entry((blk << 32) | prefix).or_default().push(i as u32);
        z.entry(blk).or_default().push(i as u32);
    }
    println!("  table {}: {} zkeys, {} time-blocks", tag, zk.len(), z.len());
    Table { dirs, mean, zkey_buckets: zk, z_buckets: z }
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

fn rerank(q: &[f32], embs: &[Vec<f32>], seen: &HashSet<u32>, k: usize) -> Vec<u32> {
    let mut s: Vec<(f32, u32)> = Vec::with_capacity(seen.len());
    for &i in seen {
        let e = &embs[i as usize];
        let mut d = 0.0f32;
        for j in 0..q.len() { d += q[j] * e[j]; }
        s.push((d, i));
    }
    s.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    s.truncate(k);
    s.into_iter().map(|(_, i)| i).collect()
}

fn main() {
    let mut rng = Rng::new(SEED);
    println!("Generating OU corpus (truth spread: mean 0.25 s, max 0.6 s)...");
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
    println!("Building {} tables per block size...", L_TABLES);
    let mut by_block: Vec<(f32, Vec<Table>)> = Vec::new();
    for &block_s in &[0.5f32, 1.0] {
        println!(" block = {} s:", block_s);
        let tabs = (0..L_TABLES).map(|t| build_table(&embs, &times, prefix_bits,
            block_s, SEED ^ (0x9E3779B97F4A7C15u64).wrapping_mul(t as u64 + 1), t))
            .collect();
        by_block.push((block_s, tabs));
    }
    let masks = probe_masks(prefix_bits, RADIUS);

    let mut queries = Vec::new();
    let mut truths = Vec::new();
    let mut srcs = Vec::new();
    for _ in 0..N_QUERIES {
        let src = (rng.next_u64() as usize) % N;
        let q: Vec<f32> = embs[src].iter()
            .map(|a| a + (rng.gauss() as f32) * QUERY_NOISE).collect();
        let mut h: BinaryHeap<Score> = BinaryHeap::new();
        for (i, e) in embs.iter().enumerate() {
            let mut s = 0.0f32;
            for j in 0..768 { s += q[j] * e[j]; }
            if h.len() < K { h.push(Score(s, i as u32)); }
            else if s > h.peek().unwrap().0 { h.pop(); h.push(Score(s, i as u32)); }
        }
        truths.push(h.into_iter().map(|Score(_, i)| i).collect::<HashSet<u32>>());
        queries.push(q);
        srcs.push(src);
    }

    println!("\n{:>6} | {:>4} | {:>10} | {:>8} | {:>8} | {:>9}",
             "mode", "blk", "recall@10", "us/q", "cand/q", "lookups/q");
    println!("{}", "-".repeat(58));
    let eval = |name: &str, blk: f32, res: &[(f64, f64, usize, usize)]| {
        let qn = N_QUERIES as f64;
        let r: f64 = res.iter().map(|x| x.0).sum::<f64>() / qn;
        let t: f64 = res.iter().map(|x| x.1).sum::<f64>() / qn;
        let c: f64 = res.iter().map(|x| x.2).sum::<usize>() as f64 / qn;
        let l: f64 = res.iter().map(|x| x.3).sum::<usize>() as f64 / qn;
        println!("{:>6} | {:>4.1} | {:>9.1}% | {:>6.0} | {:>8.0} | {:>9.0}",
                 name, blk, 100.0 * r, 1e6 * t, c, l);
    };
    let mut rng2 = Rng::new(SEED);
    for &(block_s, ref tabs) in &by_block {
        let (mut a, mut b, mut c, mut d) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for qi in 0..N_QUERIES {
            let q = &queries[qi];
            let t_q = times[srcs[qi]];
            // A: ungated
            let t0 = Instant::now();
            let mut seen: HashSet<u32> = HashSet::new();
            let mut lookups = 0usize;
            for tab in tabs {
                let prefix = sketch_of(&tab.dirs, &tab.mean, q) >> (64 - prefix_bits);
                for &m in &masks {
                    if let Some(v) = tab.zkey_buckets.get(&((u64::MAX << 32) | prefix ^ m)) {
                        lookups += 1;
                        for &i in v { seen.insert(i); }
                    }
                }
            }
            let el = t0.elapsed().as_secs_f64();
            let got = rerank(q, &embs, &seen, K);
            a.push((got.iter().filter(|i| truths[qi].contains(i)).count() as f64
                    / K as f64, el, seen.len(), lookups));
            // B: post-filter gate
            let t0 = Instant::now();
            let el = t0.elapsed().as_secs_f64();
            let mut s: Vec<(f32, u32)> = Vec::with_capacity(seen.len());
            for &i in &seen {
                if (times[i as usize] - t_q).abs() > GATE { continue; }
                let e = &embs[i as usize];
                let mut d = 0.0f32;
                for j in 0..768 { d += q[j] * e[j]; }
                s.push((d, i));
            }
            s.sort_unstable_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal));
            s.truncate(K);
            let kept = s.len();
            b.push((s.iter().filter(|(_, i)| truths[qi].contains(i)).count() as f64
                    / K as f64, el, kept, lookups));
            // C: Z-in-key gate (probe only blocks in window)
            let t0 = Instant::now();
            let mut seen_c: HashSet<u32> = HashSet::new();
            let mut lookups_c = 0usize;
            let lo = ((t_q - GATE) / block_s).floor() as u64;
            let hi = ((t_q + GATE) / block_s).floor() as u64;
            for tab in tabs {
                let prefix = sketch_of(&tab.dirs, &tab.mean, q) >> (64 - prefix_bits);
                for blk in lo..=hi {
                    for &m in &masks {
                        if let Some(v) = tab.zkey_buckets.get(&((blk << 32) | (prefix ^ m))) {
                            lookups_c += 1;
                            for &i in v { seen_c.insert(i); }
                        }
                    }
                }
            }
            let el = t0.elapsed().as_secs_f64();
            let got = rerank(q, &embs, &seen_c, K);
            c.push((got.iter().filter(|i| truths[qi].contains(i)).count() as f64
                    / K as f64, el, seen_c.len(), lookups_c));
            // D: time-range query (Z as primary key; rank window by dot product)
            let t0 = Instant::now();
            let mut seen_d: HashSet<u32> = HashSet::new();
            for tab in tabs {
                for blk in lo..=hi {
                    if let Some(v) = tab.z_buckets.get(&blk) {
                        for &i in v { seen_d.insert(i); }
                    }
                }
            }
            let el = t0.elapsed().as_secs_f64();
            let got = rerank(q, &embs, &seen_d, K);
            d.push((got.iter().filter(|i| truths[qi].contains(i)).count() as f64
                    / K as f64, el, seen_d.len(), 0));
        }
        eval("A ungate", block_s, &a);
        eval("B postflt", block_s, &b);
        eval("C zkey", block_s, &c);
        eval("D zrange", block_s, &d);
        println!("{}", "-".repeat(58));
    }
    println!("\nPredictions: C recall ~= B recall, C lookups/q << A (time prunes at probe time);");
    println!("D answers 'what is in [t-GATE, t+GATE]' — a query type the old design cannot express.");
    println!("Note: mode A's wildcard block key is a simplification; its number is the ungated reference.");
}
