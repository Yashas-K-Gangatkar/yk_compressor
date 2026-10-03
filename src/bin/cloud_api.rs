use axum::{routing::post, Router, extract::State, http::StatusCode, response::IntoResponse};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use base64::{engine::general_purpose, Engine as _};
use serde::{Serialize, Deserialize};

// ==========================================
// YK-Spatial Cloud API (Phase 3)
// POST /quantize  - Compress f32 weights to 4-bit
// POST /index     - Index embeddings in 3D grid
// POST /query     - Query the 3D grid (O(1))
// ==========================================

#[derive(Clone)]
struct AppState {
    // In-memory 3D spatial grid (for demo; production uses Redis/disk)
    grid: Arc<RwLock<HashMap<(u32, u32, u32), Vec<(usize, Vec<f32>)>>>>,
}

#[derive(Deserialize)]
struct QuantizeRequest {
    weights_b64: String, // Base64-encoded raw f32 bytes
}

#[derive(Serialize)]
struct QuantizeResponse {
    status: String,
    original_size: usize,
    packed_size: usize,
    scale: f32,
    packed_b64: String,
}

#[derive(Deserialize)]
struct IndexRequest {
    embeddings_b64: String, // Base64-encoded: [num_items, then for each: timecode(f32) + 768 floats]
    modality: u8,          // 0=Text, 1=Video, 2=Audio
}

#[derive(Serialize)]
struct IndexResponse {
    status: String,
    items_indexed: usize,
    grid_buckets: usize,
}

#[derive(Deserialize)]
struct QueryRequest {
    embedding: Vec<f32>,   // 768-dim query embedding
    timecode: f32,         // Target time
    modality: u8,          // Target modality
}

#[derive(Serialize)]
struct QueryResponse {
    status: String,
    found: bool,
    results: Vec<QueryResult>,
    query_time_ns: u64,
}

#[derive(Serialize)]
struct QueryResult {
    pair_id: usize,
    embedding_preview: [f32; 4], // First 4 values for verification
}

fn hash_embedding(emb: &[f32]) -> u32 {
    let mut hash: u32 = 0;
    for i in 0..8.min(emb.len()) {
        let scaled = (emb[i] * 10000.0) as i32;
        hash = hash.wrapping_mul(31).wrapping_add(scaled as u32);
    }
    hash
}

fn absmax_quantize_4bit(weights: &[f32]) -> (Vec<u8>, f32) {
    let max_val = weights.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
    let scale = max_val / 7.0;
    if scale == 0.0 { return (vec![0; (weights.len() + 1) / 2], 0.0); }

    let mut packed = Vec::with_capacity((weights.len() + 1) / 2);
    for chunk in weights.chunks(2) {
        let q1 = (chunk[0] / scale).round() as i8;
        let u1 = (q1 + 8) as u8 & 0x0F;
        if chunk.len() == 2 {
            let q2 = (chunk[1] / scale).round() as i8;
            let u2 = (q2 + 8) as u8 & 0x0F;
            packed.push((u1 << 4) | u2);
        } else {
            packed.push(u1 << 4);
        }
    }
    (packed, scale)
}

async fn quantize_api(State(state): State<AppState>, body: axum::body::Bytes) -> impl IntoResponse {
    let body_str = match String::from_utf8(body.to_vec()) {
        Ok(s) => s,
        Err(_) => return (StatusCode::BAD_REQUEST, "Invalid UTF-8".to_string()),
    };
    
    let req: QuantizeRequest = match serde_json::from_str(&body_str) {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_REQUEST, format!("Invalid JSON: {}", e)),
    };

    let packed_bytes = match general_purpose::STANDARD.decode(&req.weights_b64) {
        Ok(b) => b,
        Err(_) => return (StatusCode::BAD_REQUEST, "Invalid base64".to_string()),
    };

    let weights: &[f32] = unsafe {
        std::slice::from_raw_parts(packed_bytes.as_ptr() as *const f32, packed_bytes.len() / 4)
    };

    let (packed, scale) = absmax_quantize_4bit(weights);
    let b64_packed = general_purpose::STANDARD.encode(&packed);

    let response = QuantizeResponse {
        status: "success".to_string(),
        original_size: packed_bytes.len(),
        packed_size: packed.len(),
        scale,
        packed_b64: b64_packed,
    };

    (StatusCode::OK, serde_json::to_string(&response).unwrap_or_default())
}

async fn index_api(State(state): State<AppState>, body: axum::body::Bytes) -> impl IntoResponse {
    let body_str = match String::from_utf8(body.to_vec()) {
        Ok(s) => s,
        Err(_) => return (StatusCode::BAD_REQUEST, "Invalid UTF-8".to_string()),
    };

    let req: IndexRequest = match serde_json::from_str(&body_str) {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_REQUEST, format!("Invalid JSON: {}", e)),
    };

    let raw_bytes = match general_purpose::STANDARD.decode(&req.embeddings_b64) {
        Ok(b) => b,
        Err(_) => return (StatusCode::BAD_REQUEST, "Invalid base64".to_string()),
    };

    if raw_bytes.len() < 4 {
        return (StatusCode::BAD_REQUEST, "Data too short".to_string());
    }

    let num_items = u32::from_le_bytes([raw_bytes[0], raw_bytes[1], raw_bytes[2], raw_bytes[3]]) as usize;
    let item_size = 4 + (768 * 4); // timecode + 768 floats
    let mut pos = 4;

    let mut grid = state.grid.write().await;
    grid.clear();

    for i in 0..num_items {
        if pos + item_size > raw_bytes.len() { break; }

        let timecode = f32::from_le_bytes([raw_bytes[pos], raw_bytes[pos+1], raw_bytes[pos+2], raw_bytes[pos+3]]);
        pos += 4;

        let mut emb = vec![0.0f32; 768];
        for j in 0..768 {
            let off = pos + j * 4;
            emb[j] = f32::from_le_bytes([raw_bytes[off], raw_bytes[off+1], raw_bytes[off+2], raw_bytes[off+3]]);
        }
        pos += 3072;

        let h = hash_embedding(&emb);
        let coord = (h, req.modality as u32, (timecode * 10.0) as u32);
        grid.entry(coord).or_insert_with(Vec::new).push((i, emb));
    }

    let buckets = grid.len();
    let response = IndexResponse {
        status: "success".to_string(),
        items_indexed: num_items,
        grid_buckets: buckets,
    };

    (StatusCode::OK, serde_json::to_string(&response).unwrap_or_default())
}

async fn query_api(State(state): State<AppState>, body: axum::body::Bytes) -> impl IntoResponse {
    let body_str = match String::from_utf8(body.to_vec()) {
        Ok(s) => s,
        Err(_) => return (StatusCode::BAD_REQUEST, "Invalid UTF-8".to_string()),
    };

    let req: QueryRequest = match serde_json::from_str(&body_str) {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_REQUEST, format!("Invalid JSON: {}", e)),
    };

    let h = hash_embedding(&req.embedding);
    let coord = (h, req.modality as u32, (req.timecode * 10.0) as u32);

    let start = std::time::Instant::now();
    
    let grid = state.grid.read().await;
    let results_raw = grid.get(&coord).cloned().unwrap_or_default();
    
    let elapsed = start.elapsed();

    let results: Vec<QueryResult> = results_raw.iter().map(|(id, emb)| {
        QueryResult {
            pair_id: *id,
            embedding_preview: [emb[0], emb[1], emb[2], emb[3]],
        }
    }).collect();

    let response = QueryResponse {
        status: "success".to_string(),
        found: !results.is_empty(),
        results,
        query_time_ns: elapsed.as_nanos() as u64,
    };

    (StatusCode::OK, serde_json::to_string(&response).unwrap_or_default())
}

#[tokio::main]
async fn main() {
    let state = AppState {
        grid: Arc::new(RwLock::new(HashMap::new())),
    };

    let app = Router::new()
        .route("/quantize", post(quantize_api))
        .route("/index", post(index_api))
        .route("/query", post(query_api))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("YK-Spatial Cloud API running on http://localhost:3000");
    println!("Endpoints:");
    println!("  POST /quantize  - Compress f32 weights to 4-bit");
    println!("  POST /index     - Index embeddings in 3D spatial grid");
    println!("  POST /query     - O(1) spatial-temporal query");
    println!("\nExample Python usage:");
    println!("  import requests");
    println!("  r = requests.post('http://localhost:3000/quantize', json={{'weights_b64': base64.b64encode(data).decode()}})");

    axum::serve(listener, app).await.unwrap();
}
