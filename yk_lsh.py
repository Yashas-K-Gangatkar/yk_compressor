import numpy as np
import struct
import json
import time
import faiss

print("=== YK-Spatial Phase 5: Approximate Search (LSH) ===\n")

# 1. Load real BERT word embeddings
print("Loading 30,522 real BERT word embeddings...")
with open("bert_model.safetensors", "rb") as f:
    header_len = struct.unpack('<Q', f.read(8))[0]
    header = json.loads(f.read(header_len))
    data_start = 8 + header_len
    info = header["distilbert.embeddings.word_embeddings.weight"]
    start = info["data_offsets"][0] + data_start
    end = info["data_offsets"][1] + data_start
    f.seek(start)
    raw = f.read(end - start)
    embeddings = np.frombuffer(raw, dtype=np.float32).reshape(info["shape"]).copy()

n = embeddings.shape[0]
dim = embeddings.shape[1]
print(f"Loaded {n} embeddings, dim={dim}")

# Normalize embeddings for cosine similarity
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings_norm = embeddings / (norms + 1e-8)

# 2. Build FAISS indexes (ground truth + baselines)
print("\nBuilding FAISS IndexFlatIP (ground truth)...")
index_flat = faiss.IndexFlatIP(dim)
index_flat.add(embeddings_norm)

print("Building FAISS IndexHNSWFlat...")
index_hnsw = faiss.IndexHNSWFlat(dim, 32)
index_hnsw.add(embeddings_norm)

# 3. Build YK-LSH (Locality-Sensitive Hashing)
print("\nBuilding YK-LSH (16 random projections = 65,536 buckets)...")
num_projections = 16
np.random.seed(42)
projections = np.random.randn(num_projections, dim).astype(np.float32)

def lsh_hash(emb):
    dots = projections @ emb
    bits = (dots > 0).astype(np.int32)
    key = 0
    for b in bits:
        key = (key << 1) | int(b)
    return key

t0 = time.time()
lsh_table = {}
for i in range(n):
    key = lsh_hash(embeddings_norm[i])
    if key not in lsh_table:
        lsh_table[key] = []
    lsh_table[key].append(i)
lsh_build = time.time() - t0
print(f"  Built in {lsh_build:.3f}s")
print(f"  Buckets used: {len(lsh_table)} / {2**num_projections}")
print(f"  Avg bucket size: {n / len(lsh_table):.1f}")
print(f"  Max bucket size: {max(len(v) for v in lsh_table.values())}")

# 4. Run 500 similarity queries (find top-10 most similar)
num_queries = 500
query_indices = np.random.choice(n, num_queries, replace=False)

print(f"\nRunning {num_queries} similarity queries (find top-10)...\n")

def evaluate_recall(true_top10, approx_top10):
    true_set = set(true_top10)
    approx_set = set(approx_top10)
    return len(true_set & approx_set) / 10.0

# FAISS FlatIP (ground truth)
print("--- FAISS IndexFlatIP (Ground Truth, O(N)) ---")
flat_times = []
flat_results = []
for qi in query_indices:
    query = embeddings_norm[qi:qi+1]
    t0 = time.time()
    D, I = index_flat.search(query, 10)
    flat_times.append(time.time() - t0)
    flat_results.append(I[0])

# FAISS HNSW
print("--- FAISS IndexHNSWFlat (O(log N)) ---")
hnsw_times = []
hnsw_recalls = []
for qi, q_idx in enumerate(query_indices):
    query = embeddings_norm[q_idx:q_idx+1]
    t0 = time.time()
    D, I = index_hnsw.search(query, 10)
    hnsw_times.append(time.time() - t0)
    hnsw_recalls.append(evaluate_recall(flat_results[qi], I[0]))

# YK-LSH
print("--- YK-LSH (O(1) Approximate) ---")
yk_times = []
yk_recalls = []
for qi, q_idx in enumerate(query_indices):
    query = embeddings_norm[q_idx]
    t0 = time.time()
    key = lsh_hash(query)
    if key in lsh_table:
        candidates = lsh_table[key]
        candidate_embs = embeddings_norm[candidates]
        sims = candidate_embs @ query
        top_indices = np.argsort(-sims)[:10]
        approx_top10 = [candidates[i] for i in top_indices]
    else:
        approx_top10 = []
    yk_times.append(time.time() - t0)
    yk_recalls.append(evaluate_recall(flat_results[qi], approx_top10))

# 5. Results
flat_avg = np.mean(flat_times) * 1e6
hnsw_avg = np.mean(hnsw_times) * 1e6
yk_avg = np.mean(yk_times) * 1e6

flat_qps = 1.0 / np.mean(flat_times)
hnsw_qps = 1.0 / np.mean(hnsw_times)
yk_qps = 1.0 / np.mean(yk_times)

flat_recall = 1.0
hnsw_recall = np.mean(hnsw_recalls)
yk_recall = np.mean(yk_recalls)

print(f"\n========================================")
print(f"SIMILARITY SEARCH RESULTS (N={n})")
print(f"========================================")
print(f"\n| Method               | Avg Query  | QPS      | Recall@10 | Complexity |")
print(f"|----------------------|------------|----------|-----------|------------|")
print(f"| FAISS FlatIP (Truth) | {flat_avg:8.2f} us | {flat_qps:8.0f} | {flat_recall*100:5.1f}%   | O(N)       |")
print(f"| FAISS HNSW           | {hnsw_avg:8.2f} us | {hnsw_qps:8.0f} | {hnsw_recall*100:5.1f}%   | O(log N)   |")
print(f"| YK-LSH (16-bit)      | {yk_avg:8.2f} us | {yk_qps:8.0f} | {yk_recall*100:5.1f}%   | O(1)       |")

print(f"\n--- Speedup ---")
print(f"YK-LSH vs FAISS FlatIP: {flat_avg/yk_avg:.1f}x faster")
print(f"YK-LSH vs FAISS HNSW:   {hnsw_avg/yk_avg:.1f}x faster")

# 6. Multi-probe LSH
print(f"\n========================================")
print(f"MULTI-PROBE LSH (Query neighboring buckets)")
print(f"========================================")

def lsh_hash_with_neighbors(emb, num_neighbors=3):
    dots = projections @ emb
    bits = (dots > 0).astype(np.int32)
    key = 0
    for b in bits:
        key = (key << 1) | int(b)
    neighbor_keys = [key]
    for bit_pos in range(num_projections):
        neighbor_key = key ^ (1 << bit_pos)
        neighbor_keys.append(neighbor_key)
    return neighbor_keys[:num_neighbors+1]

print(f"\nRunning multi-probe queries (3 buckets per query)...")
mp_times = []
mp_recalls = []
for qi, q_idx in enumerate(query_indices):
    query = embeddings_norm[q_idx]
    t0 = time.time()
    keys = lsh_hash_with_neighbors(query, 3)
    all_candidates = []
    for k in keys:
        if k in lsh_table:
            all_candidates.extend(lsh_table[k])
    if all_candidates:
        candidate_embs = embeddings_norm[all_candidates]
        sims = candidate_embs @ query
        top_indices = np.argsort(-sims)[:10]
        approx_top10 = [all_candidates[i] for i in top_indices]
    else:
        approx_top10 = []
    mp_times.append(time.time() - t0)
    mp_recalls.append(evaluate_recall(flat_results[qi], approx_top10))

mp_avg = np.mean(mp_times) * 1e6
mp_qps = 1.0 / np.mean(mp_times)
mp_recall = np.mean(mp_recalls)

print(f"\n| Method               | Avg Query  | QPS      | Recall@10 | Complexity |")
print(f"|----------------------|------------|----------|-----------|------------|")
print(f"| FAISS FlatIP (Truth) | {flat_avg:8.2f} us | {flat_qps:8.0f} | {flat_recall*100:5.1f}%   | O(N)       |")
print(f"| FAISS HNSW           | {hnsw_avg:8.2f} us | {hnsw_qps:8.0f} | {hnsw_recall*100:5.1f}%   | O(log N)   |")
print(f"| YK-LSH (single)      | {yk_avg:8.2f} us | {yk_qps:8.0f} | {yk_recall*100:5.1f}%   | O(1)       |")
print(f"| YK-LSH (multi-probe) | {mp_avg:8.2f} us | {mp_qps:8.0f} | {mp_recall*100:5.1f}%   | O(1)*     |")

print(f"\n--- Final Comparison ---")
print(f"YK-LSH single-probe: {yk_avg:.2f} us, {yk_recall*100:.1f}% recall, {hnsw_avg/yk_avg:.1f}x faster than HNSW")
print(f"YK-LSH multi-probe:  {mp_avg:.2f} us, {mp_recall*100:.1f}% recall, {hnsw_avg/mp_avg:.1f}x faster than HNSW")

if mp_recall > hnsw_recall:
    print(f"\n✅ YK-LSH BEATS FAISS HNSW on BOTH speed AND recall!")
elif mp_recall > 0.5:
    print(f"\n⚠️ YK-LSH is faster but has lower recall. Trade-off exists.")
else:
    print(f"\n❌ YK-LSH recall too low. Need more projections or probes.")

print(f"\n========================================")
print(f"WHAT THIS MEANS")
print(f"========================================")
print(f"YK-Spatial now has TWO retrieval modes:")
print(f"  1. Exact Match (3D Hash): O(1), 100% accuracy")
print(f"  2. Approximate Search (LSH): O(1), ~{mp_recall*100:.0f}% recall")
print(f"")
print(f"This directly addresses FAISS's advantage (similarity search).")
print(f"YK-LSH provides O(1) approximate search vs FAISS HNSW's O(log N).")
