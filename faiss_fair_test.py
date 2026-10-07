# faiss_fair_test.py — FAISS on the SAME perturbed queries + SAME brute-force
# truth as stair_590k.rs. Recall@10 is the metric. No self-retrieval, no
# duplicates-dependent "accuracy". Run AFTER stair_590k (needs its output).
import numpy as np, struct, time, faiss

with open("yk_words.bin", "rb") as f:
    n = struct.unpack('I', f.read(4))[0]
X = np.fromfile("yk_words.bin", dtype=np.float32, count=1 + n * 769, offset=4).reshape(n, 769)
E = np.ascontiguousarray(X[:, 1:])
print(f"Corpus: {n} x 768 ({E.nbytes/1e9:.2f} GB)")

with open("benchmark_queries.bin", "rb") as f:
    nq, K = struct.unpack('II', f.read(8))
    raw = f.read()
print(f"Queries: {nq}, K={K}")

Q = np.empty((nq, 768), dtype=np.float32)
T = np.empty((nq, K), dtype=np.int64)
off = 0
for i in range(nq):
    Q[i] = np.frombuffer(raw, dtype=np.float32, count=768, offset=off); off += 768 * 4
    T[i] = np.frombuffer(raw, dtype=np.uint32, count=K, offset=off); off += K * 4

def recall(I):
    hits = sum(len(set(I[i]) & set(T[i])) for i in range(nq))
    return 100.0 * hits / (nq * K)

print("\nBuilding IndexFlatIP...")
t0 = time.time()
flat = faiss.IndexFlatIP(768)
flat.add(E)
print(f"  built in {time.time()-t0:.1f}s")

t0 = time.time()
D, I = flat.search(Q, K)          # batched: FAISS's intended usage
flat_ms = (time.time() - t0) * 1000
print(f"IndexFlatIP:        recall@10 = {recall(I):5.1f}%   {flat_ms/nq*1000:8.1f} us/query (batched)")

print("\nBuilding IndexHNSWFlat (M=32)... this takes a few minutes")
t0 = time.time()
hnsw = faiss.IndexHNSWFlat(768, 32)
hnsw.add(E)
print(f"  built in {time.time()-t0:.1f}s")

for ef in (16, 64):
    hnsw.hnsw.efSearch = ef
    t0 = time.time()
    hits = 0
    for i in range(nq):
        D, I = hnsw.search(Q[i:i+1], K)
        hits += len(set(I[0]) & set(T[i]))
    ms = (time.time() - t0) * 1000
    print(f"IndexHNSW ef={ef:3d}: recall@10 = {100.0*hits/(nq*K):5.1f}%   {ms/nq*1000:8.1f} us/query")
