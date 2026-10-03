use std::fs::File;
use std::io::Write;
use std::env;
use memmap2::Mmap;
use std::time::Instant;

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: cargo run --release --bin safetensors_quantize -- <input.safetensors>");
        std::process::exit(1);
    }

    let input_filename = &args[1];
    let output_filename = "model_4bit.yk";

    println!("=== YK AI: Native .safetensors 4-Bit Quantizer ===");
    println!("Loading {}...", input_filename);

    let file = File::open(input_filename)?;
    let mmap = unsafe { Mmap::map(&file)? };
    
    let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
    let data_start = 8 + header_len;
    let data_end = mmap.len();
    
    println!("Safetensors JSON header size: {} bytes", header_len);

    let weights: &[f32] = unsafe {
        std::slice::from_raw_parts(mmap[data_start..].as_ptr() as *const f32, (data_end - data_start) / 4)
    };

    let output_file = File::create(output_filename)?;
    let mut writer = std::io::BufWriter::new(&output_file);

    let start = Instant::now();
    let block_size = 256;
    let mut total_packed_bytes = 0;

    for block in weights.chunks(block_size) {
        let max_val = block.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
        let scale = max_val / 7.0;
        
        writer.write_all(&scale.to_le_bytes())?;
        total_packed_bytes += 4;

        let mut packed_bytes = Vec::with_capacity(block.len() / 2);
        for chunk in block.chunks(2) {
            let q1 = (chunk[0] / scale).round() as i8;
            let u1 = (q1 + 8) as u8 & 0x0F;
            if chunk.len() == 2 {
                let q2 = (chunk[1] / scale).round() as i8;
                let u2 = (q2 + 8) as u8 & 0x0F;
                packed_bytes.push((u1 << 4) | u2);
            } else {
                packed_bytes.push(u1 << 4);
            }
        }
        writer.write_all(&packed_bytes)?;
        total_packed_bytes += packed_bytes.len();
    }

    writer.flush()?;
    let elapsed = start.elapsed();

    println!("\nQuantization Complete!");
    println!("Original tensor size: {:.2} GB", (data_end - data_start) as f64 / (1024.0 * 1024.0 * 1024.0));
    println!("Packed size:          {:.2} GB", total_packed_bytes as f64 / (1024.0 * 1024.0 * 1024.0));
    println!("Compression:          {:.2}x reduction", (data_end - data_start) as f64 / total_packed_bytes as f64);
    println!("Time taken:           {:?}", elapsed);

    Ok(())
}
