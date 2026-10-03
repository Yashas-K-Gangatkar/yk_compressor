use std::collections::HashMap;
use std::time::Instant;

#[derive(Clone)]
struct ModalityToken {
    id: usize,
    modality: u8,
    timecode: f32,
    embedding: [u8; 4],
}

fn hash_to_3d(embedding: &[u8; 4], timecode: f32, modality: u8) -> (u32, u32, u32) {
    let x = ((embedding[0] as u32) << 16) | ((embedding[1] as u32) << 8) | (embedding[2] as u32);
    let y = modality as u32;
    let z = (timecode * 10.0) as u32;
    (x, y, z)
}

fn main() {
    let scales: Vec<(usize, &str)> = vec![
        (1_000_000, "1 Million"),
        (10_000_000, "10 Million"),
    ];

    for (n_tokens, label) in scales {
        println!("\n=== YK-Spatial: {} Token Benchmark ===", label);

        let mut grid: HashMap<(u32, u32, u32), Vec<usize>> = HashMap::new();
        let mut all_tokens: Vec<ModalityToken> = Vec::with_capacity(n_tokens);

        // Generate tokens
        let gen_start = Instant::now();
        for i in 0..n_tokens {
            let time = (i as f32) * 0.01;
            let modality = if i % 2 == 0 { 1 } else { 2 };
            let embedding = [
                ((i * 7) % 256) as u8,
                ((i * 13) % 256) as u8,
                ((i * 29) % 256) as u8,
                ((i * 37) % 256) as u8,
            ];
            let token = ModalityToken { id: i, modality, timecode: time, embedding };
            let coord = hash_to_3d(&token.embedding, token.timecode, token.modality);
            grid.entry(coord).or_insert_with(Vec::new).push(token.id);
            all_tokens.push(token);
        }
        let gen_time = gen_start.elapsed();
        println!("Generated {} tokens in {:?}", n_tokens, gen_time);
        println!("3D Grid buckets: {}", grid.len());
        println!("Memory: {:.2} MB", (n_tokens * 13) as f64 / (1024.0 * 1024.0));

        // Pick needle
        let needle_id = n_tokens / 2 + 777;
        let needle = &all_tokens[needle_id];

        // YK O(1) Query
        let yk_start = Instant::now();
        let coord = hash_to_3d(&needle.embedding, needle.timecode, needle.modality);
        let yk_results = grid.get(&coord).cloned().unwrap_or_default();
        let yk_time = yk_start.elapsed();
        let yk_found = yk_results.contains(&needle_id);

        // Standard O(N) Query
        let std_start = Instant::now();
        let mut std_found = false;
        for token in &all_tokens {
            if token.modality == needle.modality && (token.timecode - needle.timecode).abs() < 0.1 {
                if token.embedding[0] == needle.embedding[0] {
                    if token.id == needle.id { std_found = true; break; }
                }
            }
        }
        let std_time = std_start.elapsed();

        // O(N²) estimate
        let n2 = n_tokens * n_tokens;
        let per_ns = std_time.as_nanos() as f64 / n_tokens as f64;
        let est_n2 = (n2 as f64 * per_ns) / 1e9;

        let speedup = std_time.as_nanos() as f64 / yk_time.as_nanos() as f64;

        println!("\n--- Results ---");
        println!("YK O(1):    {:?} | Found: {}", yk_time, yk_found);
        println!("Std O(N):   {:?} | Found: {}", std_time, std_found);
        println!("O(N²) est:  {:.2} seconds", est_n2);
        println!("Speedup:    {:.0}x", speedup);
        println!("Reduction:  {}x fewer comparisons", n_tokens);

        println!("\n| {} | O(1) | 1 | {:?} | {} |", label, yk_time, if yk_found { "Y" } else { "N" });
        println!("| {} | O(N) | {} | {:?} | {} |", label, n_tokens, std_time, if std_found { "Y" } else { "N" });
        println!("| {} | O(N²) | {:.2}T | {:.2}s | N/A |", label, n2 as f64 / 1e12, est_n2);
    }
}
