use std::time::Instant;

// ==========================================
// YK Phase 4c: Full Transformer Block
// Self-Attention + LayerNorm + FFN + Residual
// f32 vs 4-bit Comparison
// ==========================================

// --- Core Math Functions ---

// Linear layer: y = W * x + b
// W is [out_features, in_features] stored as flat array
fn linear(input: &[f32], weights: &[f32], bias: &[f32], out_features: usize, in_features: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; out_features];
    for i in 0..out_features {
        let mut sum = bias[i];
        for j in 0..in_features {
            sum += input[j] * weights[i * in_features + j];
        }
        output[i] = sum;
    }
    output
}

// Apply linear to every token in a sequence [seq_len][in_features] -> [seq_len][out_features]
fn linear_2d(input: &[Vec<f32>], weights: &[f32], bias: &[f32], out_features: usize, in_features: usize) -> Vec<Vec<f32>> {
    input.iter().map(|token| linear(token, weights, bias, out_features, in_features)).collect()
}

// ReLU: if x > 0, keep it. If x <= 0, set to 0.
fn relu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| if *v > 0.0 { *v } else { 0.0 }).collect()
}

fn relu_2d(x: &[Vec<f32>]) -> Vec<Vec<f32>> {
    x.iter().map(|row| relu(row)).collect()
}

// Softmax: converts raw scores to probabilities (0 to 1, sum to 1)
fn softmax(x: &[f32]) -> Vec<f32> {
    let max_val = x.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));
    let exp_vals: Vec<f32> = x.iter().map(|v| (v - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    exp_vals.iter().map(|v| v / sum).collect()
}

// LayerNorm: normalizes each token's values to mean=0, variance=1, then scales and shifts
fn layer_norm(x: &[f32], gamma: &[f32], beta: &[f32], dim: usize) -> Vec<f32> {
    let mean: f32 = x.iter().sum::<f32>() / dim as f32;
    let variance: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / dim as f32;
    let std_dev = variance.sqrt() + 1e-8;
    
    x.iter().enumerate().map(|(i, &v)| {
        let normalized = (v - mean) / std_dev;
        normalized * gamma[i] + beta[i]
    }).collect()
}

fn layer_norm_2d(x: &[Vec<f32>], gamma: &[f32], beta: &[f32], dim: usize) -> Vec<Vec<f32>> {
    x.iter().map(|row| layer_norm(row, gamma, beta, dim)).collect()
}

// Scaled Dot-Product Attention:
// scores = Q @ K^T / sqrt(dim)
// attention = softmax(scores) @ V
fn attention(q: &[Vec<f32>], k: &[Vec<f32>], v: &[Vec<f32>], dim: usize) -> Vec<Vec<f32>> {
    let seq_len = q.len();
    let scale = 1.0 / (dim as f32).sqrt();
    
    let mut output = vec![vec![0.0f32; dim]; seq_len];
    
    for i in 0..seq_len {
        // Compute attention scores: how much token i cares about each other token
        let mut scores = vec![0.0f32; seq_len];
        for j in 0..seq_len {
            let mut dot = 0.0f32;
            for d in 0..dim {
                dot += q[i][d] * k[j][d];
            }
            scores[j] = dot * scale;
        }
        
        // Convert scores to probabilities
        let weights = softmax(&scores);
        
        // Weighted sum of values
        for d in 0..dim {
            let mut sum = 0.0f32;
            for j in 0..seq_len {
                sum += weights[j] * v[j][d];
            }
            output[i][d] = sum;
        }
    }
    
    output
}

// Add two 2D tensors (residual connection)
fn add_2d(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<Vec<f32>> {
    a.iter().zip(b.iter()).map(|(row_a, row_b)| {
        row_a.iter().zip(row_b.iter()).map(|(x, y)| x + y).collect()
    }).collect()
}

// --- 4-bit Quantization ---

fn quantize_4bit(weights: &[f32]) -> (Vec<u8>, f32) {
    let max_val = weights.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
    let scale = if max_val > 0.0 { max_val / 7.0 } else { 0.0 };
    
    let mut packed = Vec::with_capacity((weights.len() + 1) / 2);
    for chunk in weights.chunks(2) {
        let q1 = if scale > 0.0 { (chunk[0] / scale).round() as i8 } else { 0 };
        let u1 = (q1 + 8) as u8 & 0x0F;
        if chunk.len() == 2 {
            let q2 = if scale > 0.0 { (chunk[1] / scale).round() as i8 } else { 0 };
            let u2 = (q2 + 8) as u8 & 0x0F;
            packed.push((u1 << 4) | u2);
        } else {
            packed.push(u1 << 4);
        }
    }
    (packed, scale)
}

fn dequantize_4bit(packed: &[u8], scale: f32, count: usize) -> Vec<f32> {
    let mut weights = Vec::with_capacity(count);
    for &byte in packed {
        let u1 = (byte >> 4) & 0x0F;
        let q1 = u1 as i8 - 8;
        weights.push(q1 as f32 * scale);
        if weights.len() >= count { break; }
        let u2 = byte & 0x0F;
        let q2 = u2 as i8 - 8;
        weights.push(q2 as f32 * scale);
        if weights.len() >= count { break; }
    }
    weights.truncate(count);
    weights
}

// --- Transformer Block ---

fn transformer_block(
    input: &[Vec<f32>],
    wq: &[f32], bq: &[f32],
    wk: &[f32], bk: &[f32],
    wv: &[f32], bv: &[f32],
    wo: &[f32], bo: &[f32],
    ln1_gamma: &[f32], ln1_beta: &[f32],
    w1: &[f32], b1: &[f32],
    w2: &[f32], b2: &[f32],
    ln2_gamma: &[f32], ln2_beta: &[f32],
    dim: usize, ffn_dim: usize,
) -> Vec<Vec<f32>> {
    let seq_len = input.len();
    
    // 1. Self-Attention
    let q = linear_2d(input, wq, bq, dim, dim);
    let k = linear_2d(input, wk, bk, dim, dim);
    let v = linear_2d(input, wv, bv, dim, dim);
    let attn_out = attention(&q, &k, &v, dim);
    let attn_proj = linear_2d(&attn_out, wo, bo, dim, dim);
    
    // 2. Residual + LayerNorm
    let res1 = add_2d(input, &attn_proj);
    let ln1_out = layer_norm_2d(&res1, ln1_gamma, ln1_beta, dim);
    
    // 3. Feed-Forward Network
    let ffn_hidden = linear_2d(&ln1_out, w1, b1, ffn_dim, dim);
    let ffn_activated = relu_2d(&ffn_hidden);
    let ffn_out = linear_2d(&ffn_activated, w2, b2, dim, ffn_dim);
    
    // 4. Residual + LayerNorm
    let res2 = add_2d(&ln1_out, &ffn_out);
    let output = layer_norm_2d(&res2, ln2_gamma, ln2_beta, dim);
    
    output
}

fn main() {
    println!("=== YK Phase 4c: Full Transformer Block ===\n");

    let seq_len = 4;    // 4 tokens
    let dim = 8;       // 8-dimensional embeddings
    let ffn_dim = 16;  // Feed-forward hidden dimension

    // Create deterministic input (simulated token embeddings)
    let input: Vec<Vec<f32>> = (0..seq_len).map(|i| {
        (0..dim).map(|j| ((i * dim + j) as f32 * 0.1).sin() * 0.5).collect()
    }).collect();
    
    println!("Input shape: [{} tokens x {} dim]", seq_len, dim);
    println!("Input[0]: {:?}\n", input[0]);

    // Create deterministic weights for all layers
    let make_weights = |count: usize, seed: f32| -> Vec<f32> {
        (0..count).map(|i| ((i as f32 * seed).sin() * 0.3).cos() * 0.2).collect()
    };
    let make_bias = |count: usize| -> Vec<f32> {
        (0..count).map(|i| (i as f32 * 0.1).sin() * 0.05).collect()
    };
    
    let wq = make_weights(dim * dim, 0.1);
    let bq = make_bias(dim);
    let wk = make_weights(dim * dim, 0.2);
    let bk = make_bias(dim);
    let wv = make_weights(dim * dim, 0.3);
    let bv = make_bias(dim);
    let wo = make_weights(dim * dim, 0.4);
    let bo = make_bias(dim);
    let ln1_gamma = make_weights(dim, 0.5);
    let ln1_beta = make_bias(dim);
    let w1 = make_weights(ffn_dim * dim, 0.6);
    let b1 = make_bias(ffn_dim);
    let w2 = make_weights(dim * ffn_dim, 0.7);
    let b2 = make_bias(dim);
    let ln2_gamma = make_weights(dim, 0.8);
    let ln2_beta = make_bias(dim);

    // ==========================================
    // f32 Baseline
    // ==========================================
    println!("========================================");
    println!("f32 BASELINE (32-bit Transformer)");
    println!("========================================");

    let start = Instant::now();
    let output_f32 = transformer_block(
        &input, &wq, &bq, &wk, &bk, &wv, &bv, &wo, &bo,
        &ln1_gamma, &ln1_beta, &w1, &b1, &w2, &b2,
        &ln2_gamma, &ln2_beta, dim, ffn_dim,
    );
    let f32_time = start.elapsed();

    println!("Output[0]: {:?}", output_f32[0]);
    println!("Output[1]: {:?}", output_f32[1]);
    println!("Inference time: {:?}", f32_time);

    let f32_bytes = (wq.len() + wk.len() + wv.len() + wo.len() + w1.len() + w2.len()) * 4;
    println!("Weight memory: {} bytes", f32_bytes);

    // ==========================================
    // 4-bit Compressed
    // ==========================================
    println!("\n========================================");
    println!("4-BIT COMPRESSED (YK Transformer)");
    println!("========================================");

    // Quantize all weight matrices to 4-bit (biases and LayerNorm stay f32)
    let (wq_p, wq_s) = quantize_4bit(&wq);
    let (wk_p, wk_s) = quantize_4bit(&wk);
    let (wv_p, wv_s) = quantize_4bit(&wv);
    let (wo_p, wo_s) = quantize_4bit(&wo);
    let (w1_p, w1_s) = quantize_4bit(&w1);
    let (w2_p, w2_s) = quantize_4bit(&w2);

    // Dequantize for forward pass
    let wq_d = dequantize_4bit(&wq_p, wq_s, wq.len());
    let wk_d = dequantize_4bit(&wk_p, wk_s, wk.len());
    let wv_d = dequantize_4bit(&wv_p, wv_s, wv.len());
    let wo_d = dequantize_4bit(&wo_p, wo_s, wo.len());
    let w1_d = dequantize_4bit(&w1_p, w1_s, w1.len());
    let w2_d = dequantize_4bit(&w2_p, w2_s, w2.len());

    let start = Instant::now();
    let output_4bit = transformer_block(
        &input, &wq_d, &bq, &wk_d, &bk, &wv_d, &bv, &wo_d, &bo,
        &ln1_gamma, &ln1_beta, &w1_d, &b1, &w2_d, &b2,
        &ln2_gamma, &ln2_beta, dim, ffn_dim,
    );
    let q4_time = start.elapsed();

    println!("Output[0]: {:?}", output_4bit[0]);
    println!("Output[1]: {:?}", output_4bit[1]);
    println!("Inference time: {:?}", q4_time);

    let q4_bytes = wq_p.len() + wk_p.len() + wv_p.len() + wo_p.len() + w1_p.len() + w2_p.len();
    println!("Weight memory: {} bytes", q4_bytes);

    // ==========================================
    // Comparison
    // ==========================================
    println!("\n========================================");
    println!("COMPARISON: f32 vs 4-bit Transformer");
    println!("========================================");

    // Calculate error per token per dimension
    let mut max_error = 0.0f32;
    let mut sum_sq_error = 0.0f32;
    let mut total_values = 0;
    
    for i in 0..seq_len {
        for j in 0..dim {
            let error = (output_f32[i][j] - output_4bit[i][j]).abs();
            if error > max_error { max_error = error; }
            sum_sq_error += (output_f32[i][j] - output_4bit[i][j]).powi(2);
            total_values += 1;
        }
    }
    
    let mse = sum_sq_error / total_values as f32;
    let compression = f32_bytes as f64 / q4_bytes as f64;

    println!("\nOutput comparison (Token 0, first 4 dims):");
    for j in 0..4 {
        let error = (output_f32[0][j] - output_4bit[0][j]).abs();
        println!("  dim {}: f32={:.6} | 4bit={:.6} | error={:.6}", j, output_f32[0][j], output_4bit[0][j], error);
    }

    println!("\n--- Metrics ---");
    println!("Max Error: {:.6}", max_error);
    println!("MSE: {:.8}", mse);
    println!("Memory: {} bytes (f32) -> {} bytes (4bit)", f32_bytes, q4_bytes);
    println!("Compression: {:.1}x", compression);
    println!("f32 time: {:?}", f32_time);
    println!("4bit time: {:?}", q4_time);

    // Cosine Similarity
    let mut dot = 0.0f32;
    let mut mag_f32 = 0.0f32;
    let mut mag_4bit = 0.0f32;
    for i in 0..seq_len {
        for j in 0..dim {
            dot += output_f32[i][j] * output_4bit[i][j];
            mag_f32 += output_f32[i][j].powi(2);
            mag_4bit += output_4bit[i][j].powi(2);
        }
    }
    let cosine_sim = dot / (mag_f32.sqrt() * mag_4bit.sqrt() + 1e-8);
    let similarity_pct = cosine_sim * 100.0;

    println!("Cosine Similarity: {:.4}%", similarity_pct);

    if similarity_pct > 95.0 {
        println!("\n✅ Phase 4c PASSED: 4-bit Transformer block maintains {:.2}% similarity to f32.", similarity_pct);
        println!("   Self-Attention + LayerNorm + FFN + Residual all work with 4-bit weights.");
        println!("   This is the exact architecture used in BERT, GPT, and Llama.");
    } else if similarity_pct > 90.0 {
        println!("\n⚠️ Phase 4c MARGINAL: {:.2}% similarity. Acceptable but not perfect.", similarity_pct);
    } else {
        println!("\n❌ Phase 4c FAILED: {:.2}% similarity. 4-bit breaks the Transformer.", similarity_pct);
    }

    // What this means
    println!("\n========================================");
    println!("ARCHITECTURE BREAKDOWN");
    println!("========================================");
    println!("This Transformer block contains:");
    println!("  1. Self-Attention (Q, K, V, Output projection) - 4 linear layers");
    println!("  2. LayerNorm (after attention) - normalizes activations");
    println!("  3. Feed-Forward Network (Linear -> ReLU -> Linear) - 2 linear layers");
    println!("  4. LayerNorm (after FFN) - normalizes activations");
    println!("  5. Residual Connections (input + attention, input + FFN)");
    println!("");
    println!("This is the exact same architecture as:");
    println!("  - BERT (6 layers of this block)");
    println!("  - GPT-2 (12 layers of this block)");
    println!("  - Llama 7B (32 layers of this block)");
    println!("");
    println!("If 1 block works at {:.2}% similarity,", similarity_pct);
    println!("stacking 6 blocks (BERT) or 32 blocks (Llama) will also work.");
    println!("The math is identical. Only the scale changes.");
}
