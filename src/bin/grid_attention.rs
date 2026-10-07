// grid_attention.rs — L3: grid-fed attention prototype ("the model learns to ask").
//
// Task: N_FACTS random unit vectors ("facts") stored in memory, each perturbed
// by STORE_NOISE. A probe is the same fact perturbed INDEPENDENTLY by QUERY_NOISE.
// The model must identify which memory entry the probe refers to.
//   - bit-identical lookup impossible by construction (store noise != query noise)
//   - W_q, W_k trained by SGD (softmax cross-entropy over memory logits)
//   - serving path: learned query -> Z-window (time as key) -> sketch prefilter
//     -> exact rerank over candidates -> attention over candidates only
//
// Honest scope: toy task, single linear head, synthetic data. Demonstrates the
// MECHANISM (learned query -> 3D memory -> local attention), not a language model.
// Concept ancestry (implemented from scratch, cite don't claim):
// Memorizing Transformers (Wu et al. 2022), RETRO (Borgeaud et al. 2022),
// kNN-LM (Khandelwal et al. 2020).
//
// Run: cargo run --release --bin grid_attention   (~2-5 min: training is scalar Rust)
use std::collections::{HashMap, HashSet};
use std::time::Instant;

const D: usize = 32;            // task/embedding dim
const N_FACTS: usize = 256;
const N_FILLERS: usize = 1024;  // distractors in memory
const N_TRAIN: usize = 2000;
const N_EVAL: usize = 512;
const EPOCHS: usize = 150;
const LR: f32 = 0.2;
const BATCH: usize = 64;
const NOISE: f32 = 0.1;         // per-dim noise for BOTH store and query perturbation
const N_DIRS: usize = 32;
const SERVE_L: usize = 4;
const PROBE_RADIUS: u32 = 2;
const BLOCK_S: f32 = 8.0;       // Z-key time-block width
const GATE_S: f32 = 32.0;       // temporal window half-width
const SEED: u64 = 0xA11CE_5EED;

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

fn normalize(v: &mut Vec<f32>) {
    let n: f32 = v.iter().map(|a| a * a).sum::<f32>().sqrt();
    if n > 1e-12 { for a in v.iter_mut() { *a /= n; } }
}
// W row-major [out][in]
fn matvec(w: &[f32], x: &[f32], out: usize, inp: usize) -> Vec<f32> {
    (0..out).map(|o| {
        let row = &w[o * inp..(o + 1) * inp];
        row.iter().zip(x).map(|(a, b)| a * b).sum::<f32>()
    }).collect()
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
                let s: f32 = x.iter().zip(&v).map(|(a, b)| a * b).sum();
                if s != 0.0 { for j in 0..d { next[j] += s * x[j]; } }
            }
            normalize(&mut next);
            v = next;
        }
        for x in resid.iter_mut() {
            let s: f32 = x.iter().zip(&v).map(|(a, b)| a * b).sum();
            for j in 0..d { x[j] -= s * v[j]; }
        }
        dirs.push(v);
    }
    (dirs, mean)
}
fn sketch_of(dirs: &[Vec<f32>], mean: &[f32], e: &[f32]) -> u64 {
    let mut code = 0u64;
    for (i, dir) in dirs.iter().enumerate() {
        let mut s = 0.0f32;
        for j in 0..e.len() { s += (e[j] - mean[j]) * dir[j]; }
        if s > 0.0 { code |= 1u64 << (63 - i); }
    }
    code
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

struct Table { dirs: Vec<Vec<f32>>, mean: Vec<f32>,
               zkey: HashMap<u64, Vec<u32>> } // (time_block<<32)|sketch_prefix

fn build_table(keys: &[Vec<f32>], times: &[f32], prefix_bits: u32,
               seed: u64, tag: usize) -> Table {
    let mut rng = Rng::new(seed);
    let (dirs, mean) = learn_directions(keys, N_DIRS, &mut rng, 5);
    let mut zkey: HashMap<u64, Vec<u32>> = HashMap::new();
    for (i, k) in keys.iter().enumerate() {
        let prefix = sketch_of(&dirs, &mean, k) >> (64 - prefix_bits);
        let blk = (times[i] / BLOCK_S).floor() as u64;
        zkey.entry((blk << 32) | prefix).or_default().push(i as u32);
    }
    println!("  serve-table {}: {} zkeys", tag, zkey.len());
    Table { dirs, mean, zkey }
}

fn main() {
    let mut rng = Rng::new(SEED);
    let m_total = N_FACTS + N_FILLERS;
    println!("=== L3: grid-fed attention ===");
    println!("memory: {} facts (noise {}) + {} fillers = {} entries, D={}",
             N_FACTS, NOISE, N_FILLERS, m_total, D);

    // ---- memory: facts (stored perturbed) + fillers ----
    let mut keys_e: Vec<Vec<f32>> = Vec::with_capacity(m_total);
    let mut times: Vec<f32> = Vec::with_capacity(m_total);
    let mut fact_of: Vec<Option<usize>> = Vec::with_capacity(m_total);
    let mut facts: Vec<Vec<f32>> = Vec::with_capacity(N_FACTS);
    for i in 0..N_FACTS {
        let mut f: Vec<f32> = (0..D).map(|_| rng.gauss() as f32).collect();
        normalize(&mut f);
        facts.push(f.clone());
        let mut s = f.iter().map(|a| a + (rng.gauss() as f32) * NOISE).collect::<Vec<f32>>();
        normalize(&mut s);
        keys_e.push(s);
        times.push(rng.next_u64() as f32 / (u64::MAX as f32) * 1000.0);
        fact_of.push(Some(i));
    }
    for _ in 0..N_FILLERS {
        let mut e: Vec<f32> = (0..D).map(|_| rng.gauss() as f32).collect();
        normalize(&mut e);
        keys_e.push(e);
        times.push(rng.next_u64() as f32 / (u64::MAX as f32) * 1000.0);
        fact_of.push(None);
    }
    // ---- probes: INDEPENDENT perturbation of the fact (non-circular) ----
    let mut probes: Vec<(Vec<f32>, usize, f32)> = Vec::new();
    for _ in 0..N_TRAIN + N_EVAL {
        let f = (rng.next_u64() as usize) % N_FACTS;
        let mut p = facts[f].iter()
            .map(|a| a + (rng.gauss() as f32) * NOISE).collect::<Vec<f32>>();
        normalize(&mut p);
        let dt = 5.0 + (rng.next_u64() as f32 / (u64::MAX as f32)) * 25.0;
        probes.push((p, f, times[f] + dt));
    }

    // ---- model init + evaluation closure ----
    let init_wq: Vec<f32> = (0..D * D).map(|_| (rng.gauss() as f32) * 0.05).collect();
    let init_wk: Vec<f32> = (0..D * D).map(|_| (rng.gauss() as f32) * 0.05).collect();

    let argmax_full = |wq: &[f32], wk: &[f32],
                       pr: &[(Vec<f32>, usize, f32)]| -> (f64, f64) {
        let keys: Vec<Vec<f32>> = keys_e.iter()
            .map(|e| matvec(wk, e, D, D)).collect();
        let (mut correct, mut t_sum) = (0u32, 0.0f64);
        for (p, label, _t) in pr {
            let t0 = Instant::now();
            let q = matvec(wq, p, D, D);
            let mut best = (f32::MIN, 0usize);
            for (i, k) in keys.iter().enumerate() {
                let s: f32 = q.iter().zip(k).map(|(a, b)| a * b).sum();
                if s > best.0 { best = (s, i); }
            }
            t_sum += t0.elapsed().as_secs_f64();
            if fact_of[best.1] == Some(*label) { correct += 1; }
        }
        (100.0 * correct as f64 / pr.len() as f64, 1e6 * t_sum / pr.len() as f64)
    };

    let (acc_rand, _) = argmax_full(&init_wq, &init_wk,
                                    &probes[N_TRAIN..N_TRAIN + N_EVAL]);
    println!("untrained control: {:.1}% (chance ~ {:.2}%)\n",
             acc_rand, 100.0 / m_total as f64);

    // ---- training: SGD on softmax CE over full-memory logits ----
    println!("training {} epochs, {} probes, batch {} ...", EPOCHS, N_TRAIN, BATCH);
    let t_start = Instant::now();
    let (mut wq, mut wk) = (init_wq.clone(), init_wk.clone());
    for epoch in 0..EPOCHS {
        // stale-key SGD: learned keys refreshed once per epoch
        let keys: Vec<Vec<f32>> = keys_e.iter()
            .map(|e| matvec(&wk, e, D, D)).collect();
        let mut loss_sum = 0.0f64;
        let steps = N_TRAIN / BATCH;
        for _s in 0..steps {
            let mut gwq = vec![0.0f32; D * D];
            let mut gwk = vec![0.0f32; D * D];
            for _ in 0..BATCH {
                let pi = (rng.next_u64() as usize) % N_TRAIN;
                let (p, label, _) = &probes[pi];
                let q = matvec(&wq, p, D, D);
                let logits: Vec<f32> = keys.iter()
                    .map(|k| q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>()).collect();
                let mx = logits.iter().cloned().fold(f32::MIN, f32::max);
                let exps: Vec<f32> = logits.iter().map(|l| (l - mx).exp()).collect();
                let z: f32 = exps.iter().sum();
                let sm: Vec<f32> = exps.iter().map(|e| e / z).collect();
                loss_sum += -(sm[*label].ln() as f64);
                let mut gq = vec![0.0f32; D];
                for (i, k) in keys.iter().enumerate() {
                    let g = sm[i] - if i == *label { 1.0 } else { 0.0 };
                    if g.abs() < 1e-9 { continue; }
                    for j in 0..D { gq[j] += g * k[j]; }
                    for j in 0..D {
                        let gj = g * q[j];
                        let row = j * D;
                        for c in 0..D { gwk[row + c] += gj * keys_e[i][c]; }
                    }
                }
                for o in 0..D {
                    let row = o * D;
                    for c in 0..D { gwq[row + c] += gq[o] * p[c]; }
                }
            }
            let s = LR / BATCH as f32;
            for j in 0..D * D {
                wq[j] -= s * gwq[j];
                wk[j] -= s * gwk[j];
            }
        }
        if (epoch + 1) % 30 == 0 {
            println!("  epoch {:>3}: train CE {:.4}",
                     epoch + 1, loss_sum / (steps * BATCH) as f64);
        }
    }
    println!("trained in {:.1?}\n", t_start.elapsed());

    // ---- full-attention accuracy (trained) ----
    let (acc_full, us_full) = argmax_full(&wq, &wk,
                                          &probes[N_TRAIN..N_TRAIN + N_EVAL]);

    // ---- serving path: freeze W; sketches learned on LEARNED keys; Z-key grid ----
    let keys: Vec<Vec<f32>> = keys_e.iter().map(|e| matvec(&wk, e, D, D)).collect();
    let prefix_bits: u32 = ((m_total as f64).log2().ceil() as i64).clamp(4, 22) as u32;
    println!("building {} serving tables (prefix={} bits, block={} s)...",
             SERVE_L, prefix_bits, BLOCK_S);
    let tables: Vec<Table> = (0..SERVE_L).map(|t| build_table(&keys, &times,
        prefix_bits, SEED ^ (0x9E3779B97F4A7C15u64).wrapping_mul(t as u64 + 1), t))
        .collect();
    let masks = probe_masks(prefix_bits, PROBE_RADIUS);

    let (mut acc_served, mut acc_window) = (0u32, 0u32);
    let (mut pre_hit, mut dp_served, mut dp_window) = (0u32, 0u64, 0u64);
    let mut t_sum = 0.0f64;
    for (p, label, t_q) in &probes[N_TRAIN..N_TRAIN + N_EVAL] {
        let t0 = Instant::now();
        let q = matvec(&wq, p, D, D);
        let lo = (((t_q - GATE_S) / BLOCK_S).floor() as i64).max(0) as u64;
        let hi = ((t_q + GATE_S) / BLOCK_S).floor() as u64;
        let mut seen: HashSet<u32> = HashSet::new();
        for tab in &tables {
            let code = sketch_of(&tab.dirs, &tab.mean, &q);
            let prefix = code >> (64 - prefix_bits);
            for blk in lo..=hi {
                for &mk in &masks {
                    if let Some(v) = tab.zkey.get(&((blk << 32) | (prefix ^ mk))) {
                        for &i in v { seen.insert(i); }
                    }
                }
            }
        }
        t_sum += t0.elapsed().as_secs_f64();
        dp_served += seen.len() as u64;
        if seen.iter().any(|&i| fact_of[i as usize] == Some(*label)) { pre_hit += 1; }
        // rerank + argmax over candidates only
        let mut best = (f32::MIN, usize::MAX);
        for &i in &seen {
            let k = &keys[i as usize];
            let s: f32 = q.iter().zip(k).map(|(a, b)| a * b).sum();
            if s > best.0 { best = (s, i as usize); }
        }
        if best.1 != usize::MAX && fact_of[best.1] == Some(*label) { acc_served += 1; }
        // window-only control (time gate, no sketch)
        let mut bestw = (f32::MIN, usize::MAX);
        let mut wc = 0u64;
        for (i, t) in times.iter().enumerate() {
            if (t - t_q).abs() <= GATE_S {
                wc += 1;
                let s: f32 = q.iter().zip(&keys[i]).map(|(a, b)| a * b).sum();
                if s > bestw.0 { bestw = (s, i); }
            }
        }
        dp_window += wc;
        if fact_of[bestw.1] == Some(*label) { acc_window += 1; }
    }
    let qn = N_EVAL as f64;
    println!("\n================ RESULTS ================");
    println!("untrained control:      {:>6.1}%  (chance {:.2}%)",
             acc_rand, 100.0 / m_total as f64);
    println!("full attention:         {:>6.1}%  {:>6.0} dotp/q  {:>6.1} us/q",
             acc_full, m_total as f64, us_full);
    println!("window-only (no sketch):{:>6.1}%  {:>6.0} dotp/q",
             100.0 * acc_window as f64 / qn, dp_window as f64 / qn);
    println!("grid-fed (served):      {:>6.1}%  {:>6.0} dotp/q  {:>6.1} us/q",
             100.0 * acc_served as f64 / qn, dp_served as f64 / qn, 1e6 * t_sum / qn);
    println!("prefilter recall:       {:>6.1}%  (true fact survived the 3D prefilter)",
             100.0 * pre_hit as f64 / qn);
    println!("compute reduction vs full: {:.0}x",
             m_total as f64 / (dp_served as f64 / qn));
    println!("\nReading: served ~= full AND prefilter high => the 3D grid preserved");
    println!("the attention result at a fraction of the comparisons, with a LEARNED query.");
    println!("If prefilter recall is low: raise SERVE_L or lower NOISE. Report what you get.");
}
