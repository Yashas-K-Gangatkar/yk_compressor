use std::fs::File;
use std::io::{Read, BufReader};
use std::time::Instant;

// ==========================================
// YK AI: RAW Throughput Test (Standard 4-bit)
// Reads the ENTIRE 321MB model_4bit.yk file
// ==========================================

fn main() -> std::io::Result<()> {
    println!("=== YK AI: RAW Throughput Benchmark (Standard 4-bit) ===");

    // 1. Open file and get metadata BEFORE passing to BufReader
    let file = File::open("model_4bit.yk")?;
    let metadata = file.metadata()?;
    let file_size = metadata.len() as usize;
    
    let mut reader = BufReader::new(file);
    let input = vec![0.5f32; 256]; // Static input vector for the dot product

    let mut total_dot_product = 0.0f32;
    let mut total_weights_processed = 0usize;
    
    // Standard Block format: 4 bytes scale + 128 bytes (256 packed weights)
    let mut block_buffer = [0u8; 4 + 128]; 

    println!("Processing the entire file. No fake loops...");
    let start = Instant::now();

    loop {
        // 1. Read the 132-byte block
        if reader.read_exact(&mut block_buffer).is_err() {
            break; // End of file
        }
        
        let scale = f32::from_le_bytes([block_buffer[0], block_buffer[1], block_buffer[2], block_buffer[3]]);
        let packed_bytes = &block_buffer[4..132];

        // --- EXECUTE REAL MATH FOR THIS BLOCK ---
        let mut block_dot = 0.0f32;
        let mut weight_idx = 0;

        for &byte in packed_bytes {
            // Dequantize first 4-bit weight (High Nibble)
            let u1 = (byte >> 4) & 0x0F;
            let q1 = u1 as i8 - 8;
            let deq1 = q1 as f32 * scale;
            block_dot += input[weight_idx % 256] * deq1;
            weight_idx += 1;

            // Dequantize second 4-bit weight (Low Nibble)
            let u2 = byte & 0x0F;
            let q2 = u2 as i8 - 8;
            let deq2 = q2 as f32 * scale;
            block_dot += input[weight_idx % 256] * deq2;
            weight_idx += 1;
        }
        
        total_dot_product += block_dot;
        total_weights_processed += 256;
    }

    let elapsed = start.elapsed();
    let seconds = elapsed.as_secs_f64();

    // Calculate RAW throughput
    let weights_per_sec = (total_weights_processed as f64 / seconds) / 1_000_000.0;
    let mb_per_sec = (file_size as f64 / (1024.0 * 1024.0)) / seconds;

    println!("\n=== RAW REAL-WORLD RESULTS ===");
    println!("Total File Processed:    {:.2} MB", file_size as f64 / (1024.0 * 1024.0));
    println!("Total Weights Processed: {} million", total_weights_processed / 1_000_000);
    println!("Total Time:              {:.4} seconds", seconds);
    println!("\n--- True Throughput ---");
    println!("True Processing Speed:   {:.2} Million Weights / sec", weights_per_sec);
    println!("True File I/O Speed:     {:.2} MB / sec", mb_per_sec);
    println!("Final Dot Product Sum:   {:.4}", total_dot_product);

    Ok(())
}
