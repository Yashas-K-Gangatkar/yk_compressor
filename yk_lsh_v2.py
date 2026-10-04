import numpy as np
import struct
import json
import time
import faiss

print("=== YK-Spatial Phase 5v2: Multi-Table LSH ===\n")

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

# Normalize for cosine similarity
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings_norm = embeddings / (norms + 1e-8)

# 2. Build FAISS (ground truth + HNSW)
print("Building FAISS indexes...")
index_flat = faiss.IndexFlatIP(dim)
index_flat.add(embeddings_norm)
index_hnsw = faiss.IndexHNSWFlat(dim, 32)
index_hnsw.add(embeddings_norm)

# 3. Build Multi-Table LSH
L_TABLES = 10  # Number of hash tables
K_PROJECTIONS = 12  # Projections per table (2^12 = 4096 buckets per table)

print(f"\nBuilding Multi-Table LSH: {L_TABLES} tables x {K_PROJECTIONS} projections")
print(f"Buckets per table: {2**K_PROJECTIONS}")
print(f"Expected recall: ~70% (vs 14% with single table)\n")

np.random.seed(42)
all_projections = []
all_tables = []

t0 = time.time()
for table_idx in range(L_TABLES):
    # Random projections for this table
    projections = np.random.randn(K_PROJECTIONS, dim).astype(np.float32)
    all_projections.append(projections)
    
    # Build hash table
    table = {}
    for i in range(n):
        dots = projections @ embeddings_norm[i]
        bits = (dots > 0).astype(np.int32)
        key = 0
        for b in bits:
            key = (key << 1) | int(b)
        if key not in table:
            table[key] = []
        table[key].append(i)
    
    all_tables.append(table)

lsh_build = time.time() - t0
total_buckets = sum(len(t) for t in all_tables)
print(f"  Built {L_TABLES} tables in {lsh_build:.3f}s")
print(f"  Total buckets across all tables: {total_buckets}")

# 4. Query function
def lsh_query(query_emb, k=10):
    """Query all L tables, merge candidates, rerank."""
    candidates = set()
    for table_idx in range(L_TABLES):
        projections = all_projections[table_idx]
        dots = projections @ query_emb
        bits = (dots > 0).astype(np.int32)
        key = 0
        for b in bits:
            key = (key << 1) | int(b)
        
        if key in all_tables[table_idx]:
            for c in all_tables[table_idx][key]:
                candidates.add(c)
    
    if not candidates:
        return []
    
    # Rerank by cosine similarity
    candidates = list(candidates)
    candidate_embs = embeddings_norm[candidates]
    sims = candidate_embs @ query_emb
    
    # Get top-k
    top_indices = np.argsort(-sims)[:k]
    return [candidates[i] for i in top_indices]

# 5. Run 500 queries
num_queries = 500
query_indices = np.random.choice(n, num_queries, replace=False)

print(f"\nRunning {num_queries} similarity queries...\n")

def evaluate_recall(true_top10, approx_top10):
    return len(set(true_top10) & set(approx_top10)) / 10.0

# FAISS FlatIP (ground truth)
print("--- Ground Truth (FAISS FlatIP) ---")
flat_times = []
flat_results = []
for qi in query_indices:
    query = embeddings_norm[qi:qi+1]
    t0 = time.time()
    D, I = index_flat.search(query, 10)
    flat_times.append(time.time() - t0)
    flat_results.append(I[0])

# FAISS HNSW
print("--- FAISS HNSW ---")
hnsw_times = []
hnsw_recalls = []
for qi, q_idx in enumerate(query_indices):
    query = embeddings_norm[q_idx:q_idx+1]
    t0 = time.time()
    D, I = index_hnsw.search(query, 10)
    hnsw_times.append(time.time() - t0)
    hnsw_recalls.append(evaluate_recall(flat_results[qi], I[0]))

# YK Multi-Table LSH
print("--- YK Multi-Table LSH ---")
yk_times = []
yk_recalls = []
candidate_counts = []
for qi, q_idx in enumerate(query_indices):
    query = embeddings_norm[q_idx]
    t0 = time.time()
    approx_top10 = lsh_query(query, 10)
    yk_times.append(time.time() - t0)
    yk_recalls.append(evaluate_recall(flat_results[qi], approx_top10))
    candidate_counts.append(len(approx_top10))

# 6. Results
flat_avg = np.mean(flat_times) * 1e6
hnsw_avg = np.mean(hnsw_times) * 1e6
yk_avg = np.mean(yk_times) * 1e6

hnsw_recall = np.mean(hnsw_recalls)
yk_recall = np.mean(yk_recalls)
avg_candidates = np.mean(candidate_counts)

print(f"\n========================================")
print(f"MULTI-TABLE LSH RESULTS (N={n})")
print(f"========================================")
print(f"\n| Method               | Avg Query  | QPS      | Recall@10 | Complexity |")
print(f"|----------------------|------------|----------|-----------|------------|")
print(f"| FAISS FlatIP (Truth) | {flat_avg:8.2f} us | {1e6/flat_avg:8.0f} | 100.0%   | O(N)       |")
print(f"| FAISS HNSW           | {hnsw_avg:8.2f} us | {1e6/hnsw_avg:8.0f} | {hnsw_recall*100:5.1f}%   | O(log N)   |")
print(f"| YK-LSH (single, K=16)|    74.64 us |    13397 |  14.2%   | O(1)       |")
print(f"| YK-LSH (multi, K=12) | {yk_avg:8.2f} us | {1e6/yk_avg:8.0f} | {yk_recall*100:5.1f}%   | O(1)*      |")

print(f"\n--- Speedup vs FAISS HNSW ---")
print(f"YK-LSH (single, old):  0.9x (SLOWER)")
print(f"YK-LSH (multi, new):   {hnsw_avg/yk_avg:.1f}x")

print(f"\n--- Recall Comparison ---")
print(f"FAISS HNSW:        {hnsw_recall*100:.1f}%")
print(f"YK-LSH (old, 1 table, K=16): 14.2%")
print(f"YK-LSH (new, {L_TABLES} tables, K={K_PROJECTIONS}): {yk_recall*100:.1f}%")
print(f"Improvement: +{(yk_recall - 0.142)*100:.1f} percentage points")

print(f"\n--- Candidate Analysis ---")
print(f"Avg candidates per query: {avg_candidates:.1f}")
print(f"Avg reranking cost: {avg_candidates:.0f} dot products")

if yk_recall > hnsw_recall:
    print(f"\n✅ YK-LSH BEATS FAISS HNSW on BOTH speed AND recall!")
elif yk_recall > 0.5:
    print(f"\n⚠️ YK-LSH recall improved to {yk_recall*100:.1f}% (from 14.2%)")
    print(f"   FAISS HNSW recall: {hnsw_recall*100:.1f}%")
    if yk_avg < hnsw_avg:
        print(f"   YK is {hnsw_avg/yk_avg:.1f}x faster but recall is {hnsw_recall*100 - yk_recall*100:.1f}% lower")
    else:
        print(f"   YK is slower. Need fewer projections or more tables.")
else:
    print(f"\n❌ YK-LSH recall still low at {yk_recall*100:.1f}%")
    print(f"   Need more tables (L=20+) or fewer projections (K=10)")

# 7. Try different configurations
print(f"\n========================================")
print(f"CONFIGURATION SWEEP")
print(f"========================================")

configs = [
    (5, 10, "L=5, K=10 (1024 buckets)"),
    (10, 12, "L=10, K=12 (4096 buckets)"),
    (20, 14, "L=20, K=14 (16384 buckets)"),
    (30, 16, "L=30, K=16 (65536 buckets)"),
]

print(f"\n| Config | Avg Query | Recall@10 | vs HNSW Speed | vs HNSW Recall |")
print(f"|--------|------------|-----------|---------------|----------------|")

for L, K, name in configs:
    # Build tables
    np.random.seed(42)
    tables = []
    projections_list = []
    
    for _ in range(L):
        proj = np.random.randn(K, dim).astype(np.float32)
        projections_list.append(proj)
        table = {}
        for i in range(n):
            dots = proj @ embeddings_norm[i]
            bits = (dots > 0).astype(np.int32)
            key = 0
            for b in bits:
                key = (key << 1) | int(b)
            if key not in table:
                table[key] = []
            table[key].append(i)
        tables.append(table)
    
    # Query
    times = []
    recalls = []
    for qi_idx, q_idx in enumerate(query_indices[:200]):  # 200 queries for speed
        query = embeddings_norm[q_idx]
        t0 = time.time()
        
        candidates = set()
        for t_idx in range(L):
            proj = projections_list[t_idx]
            dots = proj @ query
            bits = (dots > 0).astype(np.int32)
            key = 0
            for b in bits:
                key = (key << 1) | int(b)
            if key in tables[t_idx]:
                for c in tables[t_idx][key]:
                    candidates.add(c)
        
        if candidates:
            candidates = list(candidates)
            cand_embs = embeddings_norm[candidates]
            sims = cand_embs @ query
            top_idx = np.argsort(-sims)[:10]
            approx = [candidates[i] for i in top_idx]
        else:
            approx = []
        
        times.append(time.time() - t0)
        recalls.append(evaluate_recall(flat_results[qi_idx], approx))
    
    avg_time = np.mean(times) * 1e6
    avg_recall = np.mean(recalls)
    speed = hnsw_avg / avg_time if avg_time > 0 else 0
    recall_diff = avg_recall - hnsw_recall
    
    print(f"| {name:30s} | {avg_time:8.2f} us | {avg_recall*100:5.1f}%   | {speed:5.1f}x         | {recall_diff*100:+5.1f}%        |")

print(f"\n========================================")
print(f"WHY THE OLD LSH FAILED (EXPLANATION)")
print(f"========================================")
print(f"")
print(f"Old config: L=1 table, K=16 projections")
print(f"  Buckets: 65,536")
print(f"  Avg bucket size: {n/3822:.1f}")
print(f"  Max bucket size: 1163 (THIS WAS THE BOTTLENECK)")
print(f"  Recall: 14.2%")
print(f"")
print(f"Problem 1: Single table = low recall")
print(f"  With 1 table, similar embeddings have only ~5% chance")
print(f"  of landing in the same bucket. 95% of similar pairs are missed.")
print(f"")
print(f"Problem 2: Max bucket = 1163")
print(f"  One bucket had 1163 embeddings. Reranking 1163 candidates")
print(f"  took longer than the hash lookup itself. O(1) became O(N).")
print(f"")
print(f"Problem 3: Multi-probe only flipped 1 bit")
print(f"  Similar embeddings often differ by 2+ bits. Flipping 1 bit")
print(f"  only found nearby buckets, not the right bucket.")
print(f"")
print(f"Fix: Multiple tables (L=10+) catches pairs that 1 table misses.")
print(f"     Each table uses different random projections, so similar")
print(f"     embeddings have independent chances of collision.")
