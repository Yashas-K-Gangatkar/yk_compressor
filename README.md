# YK Spatial Memory — a two-tier 3D index with time as a first-class key

Exact O(1) hash lookup (semantic, modality, time-bucket) + STAIR: a
sign-sketch approximate tier where the Z-axis is a real key dimension —
probe-time temporal gating, time-range queries, and learned probes that
feed attention over retrieved candidates.

Honest positioning: this is a retrieval/memory index and a mechanism
study. It does not replace attention. At pure approximate search, FAISS
HNSW beats our sketch tier and we report that. Contributions: time-as-key
semantics, the measured sigma-cliff motivating two tiers, a temporal-gate
boundary condition, a grid-fed attention prototype, and a post-mortem of
nine benchmark pitfalls we committed ourselves.

## Measured results (all reproducible from this repo)

| Result | Number | Code |
|---|---|---|
| Exact tier sigma-cliff | 100% to 0% at any noise>0 | src/bin/stair_benchmark.rs |
| STAIR sweep knee (n=200) | 93.3% recall@10 @ 76 us (L=4, r=2) | src/bin/stair_sweep.rs |
| Corpus scale (589,933 embeddings) | STAIR 58.3% @ 2,260 us vs HNSW-IP 74.3% @ 109 us (HNSW wins) | src/bin/stair_590k.rs, faiss_hnsw_ip.py |
| Temporal gate, time-local corpus | 98.5% recall @ 4.7x speedup; harmful on non-temporal data | src/bin/trajectory_test.rs |
| Z-key probe-time gating | 100% recall, 4.5x faster, 28x fewer candidates | src/bin/zkey_test.rs |
| Time-range queries | new query type, 100% recall @ 2-4 us | src/bin/zkey_test.rs |
| Grid-fed attention (learned probe) | 0.0% to 94.9%, 120x fewer comparisons vs full attention | src/bin/grid_attention.rs |
| Grid-fed at D=768 (590K real) | sketch degrades gracefully (94.5->42.5% recall) where exact tier is 0%; served 42% top-1 @ 494x fewer comparisons; adapted queries break sketches unless geometry preserved | grid_attention_768.py |

## Documents

- PAPER.md — the paper (draft v1): design, measurements, post-mortem.
- HONESTY.md — what each benchmark actually measures. Read before citing.

## The nine pitfalls (post-mortem highlights)

Duplicate "corpora" from bounded image sources / self-retrieval circularity /
exact-match scoring of similarity engines / check asymmetry / IP-L2 metric
mismatch (14.2% vs 74.3% on identical queries) / cross-task speedups /
language constants / projected baselines as measurements / results without
surviving source. Each demonstrated with a measured example in PAPER.md s9.

## Status ladder

- [x] L1 — 3D-keyed exact storage
- [x] L2 — Z-key grid: probe-time gating + time-range queries
- [x] L2c — paper v1
- [ ] L2b — ann-benchmarks port
- [ ] L3 scale-up — learned probes at real dimensionality, real corpora
- [ ] L4 — hardware 3D (awareness only; not claimed)

## Build

    cargo run --release --bin stair_benchmark -- real_200.bin
    cargo run --release --bin grid_attention
    python3 faiss_hnsw_ip.py   # after stair_590k writes benchmark_queries.bin
