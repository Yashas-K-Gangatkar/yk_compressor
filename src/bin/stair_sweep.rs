// stair_sweep.rs — recall vs (tables L, probe radius) trade-off sweep.
// Same perturbed-query protocol as stair_benchmark (recall@10 vs brute-force truth).
// IMPORTANT: at n=200 brute force is also ~microseconds. This measures the
// trade-off SHAPE, not speed. Speed claims require n >= 13K.
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::io::Read;
use std::time::Instant;

const N_DIRS: usize = 32;
const K: usize = 10;
const N_QUERIES: usize = 100;
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

fn build_table(sample: &[Vec<f32>], embs: &[Vec<f32>], prefix_bits: u32,
               seed: u64) -> Table {
    let mut rng = Rng::new(seed);
    let (dirs, mean) = learn_directions(sample, N_DIRS, &mut rng, 5);
    let mut buckets: HashMap<u64, Vec<u32>> = HashMap::new();
    for (i, e) in embs.iter().enumerate() {
        let code = sketch_of(&dirs, &mean, e);
        buckets.entry(code >> (64 - prefix_bits)).or_default().push(i as u32);
    }
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
    File::open("real_200.bin").expect("real_200.bin").read_to_end(&mut data).unwrap();
    let mut pos = 0;
    let n = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    pos += 4;
    let mut embs = Vec::with_capacity(n);
    let mut times = Vec::with_capacity(n);
    for _ in 0..n {
        times.push(f32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()));
        pos += 4;
        let mut e = vec![0.0f32; 768];
        for j in 0..768 {
            let off = pos + j * 4;
            e[j] = f32::from_le_bytes(data[off..off + 4].try_into().unwrap());
        }
        pos += 3072;
        embs.push(e);
    }
    println!("Loaded {} embeddings", n);

    let prefix_bits: u32 = ((n as f64).log2().ceil() as i64).clamp(4, 22) as u32;
    println!("prefix={} bits", prefix_bits);

    let sample: Vec<Vec<f32>> = embs.clone();
    println!("Building 8 tables (different learned directions)...");
    let t0 = Instant::now();
    let tables: Vec<Table> = (0..8).map(|t|
        build_table(&sample, &embs, prefix_bits,
                    SEED ^ (0x9E3779B97F4A7C15u64).wrapping_mul(t as u64 + 1))
    ).collect();
    println!("Built in {:.2?}\n", t0.elapsed());

    let masks_r1 = probe_masks(prefix_bits, 1);
    let masks_r2 = probe_masks(prefix_bits, 2);
    let masks_r3 = probe_masks(prefix_bits, 3);

    let mut rng = Rng::new(SEED);

    // per-dimension std for noise scaling
    let mut sum = vec![0.0f32; 768];
    let mut sumsq = vec![0.0f32; 768];
    for e in &embs { for j in 0..768 { sum[j] += e[j]; sumsq[j] += e[j] * e[j]; } }
    let dim_std: Vec<f32> = (0..768).map(|j| {
        let m = sum[j] / n as f32;
        ((sumsq[j] / n as f32) - m * m).max(0.0).sqrt()
    }).collect();

    // ---- generate the sigma=0.25 query set ONCE; truth computed once ----
    let sigma = 0.25f32;
    let mut queries = Vec::with_capacity(N_QUERIES);
    let mut truths = Vec::with_capacity(N_QUERIES);
    for _ in 0..N_QUERIES {
        let src = (rng.next_u64() as usize) % n;
        let q: Vec<f32> = embs[src].iter().enumerate()
            .map(|(j, a)| a + (rng.gauss() as f32) * sigma * dim_std[j]).collect();
        let mut h: BinaryHeap<Score> = BinaryHeap::new();
        for (i, e) in embs.iter().enumerate() {
            let mut s = 0.0f32;
            for j in 0..768 { s += q[j] * e[j]; }
            if h.len() < K { h.push(Score(s, i as u32)); }
            else if s > h.peek().unwrap().0 { h.pop(); h.push(Score(s, i as u32)); }
        }
        truths.push(h.into_iter().map(|Score(_, i)| i).collect::<HashSet<u32>>());
        queries.push(q);
    }

    println!("Sweep at sigma={}: {} queries, recall@{} vs brute-force truth", sigma, N_QUERIES, K);
    println!("{:>2} | {:>6} | {:>10} | {:>9} | {:>7}", "L", "radius", "recall@10", "us/q", "cand/q");
    println!("{}", "-".repeat(45));

    for &radius in &[1u32, 2, 3] {
        let masks: &[u64] = match radius { 1 => &masks_r1, 2 => &masks_r2, _ => &masks_r3 };
        for &L in &[1usize, 2, 4, 8] {
            let mut rec = 0.0f64;
            let mut total_time = 0.0f64;
            let mut cand_sum = 0usize;
            for (qi, q) in queries.iter().enumerate() {
                let t0 = Instant::now();
                // union candidates across first L tables
                let mut seen: HashSet<u32> = HashSet::new();
                for table in &tables[..L] {
                    let code = sketch_of(&table.dirs, &table.mean, q);
                    let prefix = code >> (64 - prefix_bits);
                    for &m in masks {
                        if let Some(idxs) = table.buckets.get(&(prefix ^ m)) {
                            for &i in idxs { seen.insert(i); }
                        }
                    }
                }
                let cand = seen.len();
                let mut scored: Vec<(f32, u32)> = Vec::with_capacity(cand);
                for &i in &seen {
                    let e = &embs[i as usize];
                    let mut s = 0.0f32;
                    for j in 0..768 { s += q[j] * e[j]; }
                    scored.push((s, i));
                }
                scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                total_time += t0.elapsed().as_secs_f64();
                cand_sum += cand;
                let hits = scored.iter().take(K)
                    .filter(|(_, i)| truths[qi].contains(i)).count();
                rec += hits as f64 / K as f64;
            }
            let qn = N_QUERIES as f64;
            println!("{:>2} | {:>6} | {:>9.1}% | {:>7.2} | {:>7.1}",
                     L, radius, 100.0 * rec / qn, 1e6 * total_time / qn,
                     cand_sum as f64 / qn);
        }
    }

    println!("\nReading: recall should climb with L (and radius), time should climb too.");
    println!("The curve's knee — recall per microsecond — is the operating point.");
    println!("At n=200, brute force is also ~microseconds: this is the trade-off SHAPE, not a speed claim.");
}
