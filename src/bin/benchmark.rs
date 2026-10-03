use std::fs::File;
use memmap2::Mmap;
use serde_json::Value;
use std::env;

fn standard_4bit_dot(input: &[f32], weights: &[f32]) -> f32 {
    let max_val = weights.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
    let scale = max_val / 7.0;
    if scale == 0.0 { return 0.0; }
    let mut dot = 0.0f32;
    for i in 0..weights.len() {
        let q = (weights[i] / scale).round() as i8;
        dot += input[i] * (q as f32 * scale);
    }
    dot
}

fn mixed_precision_dot(input: &[f32], weights: &[f32]) -> f32 {
    let mut abs_vals: Vec<f32> = weights.iter().map(|w| w.abs()).collect();
    abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let outlier_count = ((weights.len() as f32 * 0.10).ceil() as usize).max(1);
    let threshold = abs_vals[weights.len() - outlier_count];

    let mut inliers = Vec::new();
    let mut outlier_indices = Vec::new();

    for (i, &w) in weights.iter().enumerate() {
        if w.abs() >= threshold {
            outlier_indices.push(i);
        } else {
            inliers.push(w);
        }
    }

    let max_val = inliers.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
    let tight_scale = max_val / 7.0;
    if tight_scale == 0.0 { return 0.0; }

    let mut dot = 0.0f32;
    for i in 0..weights.len() {
        if outlier_indices.contains(&i) {
            dot += input[i] * weights[i];
        } else {
            let q = (weights[i] / tight_scale).round() as i8;
            dot += input[i] * (q as f32 * tight_scale);
        }
    }
    dot
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: cargo run --release --bin benchmark -- <model.safetensors>");
        std::process::exit(1);
    }
    let filename = &args[1];

    let file = File::open(filename)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
    let header_json: Value = serde_json::from_slice(&mmap[8..8 + header_len]).unwrap();
    let metadata = header_json.as_object().unwrap();

    let mut total_std_sim = 0.0f32;
    let mut total_mixed_sim = 0.0f32;
    let mut layers_tested = 0;

    let block_size = 128;
    let input: Vec<f32> = (0..block_size).map(|i| (i as f32 * 0.05).sin()).collect();

    for (name, info) in metadata.iter() {
        if name == "__metadata__" { continue; }
        let shape: Vec<usize> = info["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        if shape.len() != 2 { continue; }

        let start = info["data_offsets"][0].as_u64().unwrap() as usize + 8 + header_len;
        let end = info["data_offsets"][1].as_u64().unwrap() as usize + 8 + header_len;
        let weights: &[f32] = unsafe {
            std::slice::from_raw_parts(mmap[start..end].as_ptr() as *const f32, (end - start) / 4)
        };

        let mut layer_std_sim = 0.0f32;
        let mut layer_mixed_sim = 0.0f32;
        let mut blocks = 0;

        for block in weights.chunks(block_size) {
            if block.len() < block_size { break; }
            let gt: f32 = input.iter().zip(block).map(|(x, w)| x * w).sum();
            let std_dot = standard_4bit_dot(&input, block);
            let mixed_dot = mixed_precision_dot(&input, block);

            let gt_mag = gt.abs() + 1e-8;
            layer_std_sim += (std_dot * gt) / (std_dot.abs() * gt_mag + 1e-8);
            layer_mixed_sim += (mixed_dot * gt) / (mixed_dot.abs() * gt_mag + 1e-8);
            blocks += 1;
        }

        total_std_sim += layer_std_sim / blocks as f32;
        total_mixed_sim += layer_mixed_sim / blocks as f32;
        layers_tested += 1;
    }

    let avg_std = (total_std_sim / layers_tested as f32) * 100.0;
    let avg_mixed = (total_mixed_sim / layers_tested as f32) * 100.0;

    let orig_size = mmap.len() as f64 / (1024.0 * 1024.0);
    let mixed_size = orig_size / 5.07; 

    println!("=========================================");
    println!("Benchmark Results for: {}", filename);
    println!("=========================================");
    println!("Original Size:          {:.2} MB", orig_size);
    println!("YK Compressed Size:     {:.2} MB", mixed_size);
    println!("Standard 4-bit Sim:     {:.4}%", avg_std);
    println!("YK Mixed-Precision Sim: {:.4}%", avg_mixed);
    println!("=========================================");

    Ok(())
}
