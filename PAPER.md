# Time as a First-Class Key: A Two-Tier 3D Memory Index with Learned Probes, and a Post-Mortem of Nine Benchmark Pitfalls

Yashas K — Independent Researcher, Bangalore, India
Code: https://github.com/Yashas-K-Gangatkar/yk_compressor

## Abstract

We present a two-tier memory index for multimodal AI systems. Tier 1 is an exact O(1) hash keyed on (semantic digest, modality, time bucket). Tier 2 is an approximate similarity tier — STAIR (Spatio-Temporal Approximate Index with exact Rerank) — built from data-dependent sign sketches, multi-probe bucket search, and exact rerank, in which time (Z) is a first-class key dimension: temporal constraints prune candidates at probe time, and pure time-range queries are expressible. We measure, on real and controlled corpora: (i) the σ-cliff — an exact-match tier falls from 100% to 0% recall under any input perturbation; (ii) recall/latency trade-off curves for the sketch tier; (iii) an honest corpus-scale comparison in which FAISS HNSW (inner-product metric) is superior for pure approximate search (74.3% recall@10 at 109 µs vs our best 58.3% at 2,260 µs on 589,933 embeddings); (iv) a boundary condition for temporal gating (gating preserves 98.5% recall at 4.7× speedup when ground truth is temporally concentrated, and is harmful when it is not); (v) Z-key serving (100% recall at 4.5× speedup and 28× fewer candidates, plus time-range queries); and (vi) a grid-fed attention prototype in which a learned query (trained 0.0% → 94.9% on a 1,280-entry memory) retrieves through the 3D prefilter at 120× fewer comparisons with accuracy limited only by prefilter recall. At real dimensionality (D=768, 589,933 real embeddings) the sketch tier degrades gracefully with input noise (94.5% to 42.5% prefilter recall) where the exact tier is at 0%, and we identify a co-design constraint: discriminatively adapted queries break sketch prefiltration unless query geometry is preserved (Section 8.1). We additionally document nine benchmark pitfalls — each demonstrated with a measured example from our own earlier, flawed evaluation — that silently corrupt published retrieval comparisons.

## 1. Introduction

Long-context multimodal models face a compute wall: attention over N tokens costs O(N²). A common proposed escape is retrieval: fetch the few relevant tokens instead of attending to all. Retrieval systems, however, are usually evaluated in ways that flatter them — self-retrieval, duplicate-laden corpora, mismatched metrics — and are usually keyless with respect to time.

This paper makes three contributions of different kinds:

1. **An architecture:** a two-tier 3D memory index where time is a key, not metadata — enabling probe-time temporal gating and time-range retrieval alongside exact and similarity lookup (Sections 2, 6, 7).
2. **A mechanism demonstration:** grid-fed attention — a *learned* query, trained end-to-end, retrieving through the 3D prefilter and attending locally (Section 8). This is the first non-circular retrieval result in this research program: the query is a different perturbation of the stored fact, so bit-identical lookup is impossible by construction.
3. **A post-mortem:** nine pitfalls we ourselves committed in an earlier version of this work, each reproduced and measured (Section 9). We publish them because they are common, silent, and rarely documented with numbers.

**Claims boundary.** We do *not* claim to replace attention, to beat HNSW at approximate search, or to have billion-scale results. One earlier draft by the author (never submitted) claimed all three; Section 9 retracts those claims with root causes.

## 2. Design

### 2.1 Tier 1 — exact (L1)
A hash map keyed on (semantic digest, modality, time bucket). The semantic digest is a base-31 multiplicative hash of the first 8 embedding dimensions scaled by 10⁴. Retrieval is O(1) and bit-exact. Its failure mode — measured in Section 3 — is total: any perturbation of the input changes the digest.

### 2.2 Tier 2 — approximate (STAIR)
Four stages:

1. **Sketch.** Each embedding is projected onto K data-dependent directions (power iteration with sequential deflation, learned on a sample) and encoded as K sign bits. Data-dependent directions concentrate sign information where the data varies, unlike random projections.
2. **Z-key bucketing.** The bucket key is `(time_block, top-P sketch bits)`. Time is part of the key: buckets are space-time cells (L2 of our ladder).
3. **Multi-probe, multi-table lookup.** A query probes its bucket plus all buckets within Hamming radius r of its prefix, across L tables with independently learned directions; temporal constraints select which time blocks are probed at all (**probe-time gating** — pruning happens before any similarity computation).
4. **Exact rerank.** Surviving candidates are scored by true dot product; top-k returned. The sketch never decides the answer.

Two further query modes fall out of the design: **gated similarity** (candidates restricted to a time window) and **time-range ranking** (Z as primary key — rank everything in [t₁, t₂] by similarity), which similarity-only indexes cannot express.

### 2.3 Terminology
Ours: σ-cliff, STAIR, Z-key, probe-time gating, time-range ranking, grid-fed attention, learned probe. Ancestry (concepts studied, implementations from scratch): SimHash [1], multi-probe LSH [2], HNSW [3], FAISS [4], product quantization [5]; for Section 8: Memorizing Transformers [6], RETRO [7], kNN-LM [8].

## 3. The σ-cliff

On 200 real ViT embeddings (google/vit-base-patch16-224), queries were perturbed copies of stored embeddings at noise level σ (per-dimension, in units of each dimension's standard deviation):

| σ | Tier-1 hit@1 | STAIR recall@10 |
|---|---|---|
| 0.00 | 100% | 66.6% |
| 0.10 | **0%** | 67.6% |
| 0.25 | **0%** | 65.6% |
| 0.50 | **0%** | 63.8% |
| 1.00 | **0%** | 69.4% |

The exact tier returns *nothing* under any perturbation: a semantic hash is an identity test, not a similarity function. This single table motivates the two-tier design. (Methodological note: at n=200 an early fixed-prefix configuration was candidate-starved — ~4 candidates/query capped recall near 34%; an adaptive prefix of ⌈log₂ n⌉ bits restored ~30 candidates/query and ~66% recall. We report the diagnosis because it generalizes: recall ceilings have causes, and instrumentation should say which.)

## 4. Trade-off sweep (n=200, σ=0.25, 100 queries)

| L | radius | recall@10 | µs/q | cand/q |
|---|---|---|---|---|
| 1 | 1 | 37.9% | 15.3 | 11 |
| 2 | 1 | 56.0% | 28.6 | 18 |
| 4 | 1 | 72.5% | 52.5 | 27 |
| 8 | 1 | 82.8% | 97.6 | 37 |
| 1 | 2 | 69.1% | 26.1 | 34 |
| 2 | 2 | 85.7% | 46.2 | 52 |
| **4** | **2** | **93.3%** | **76.1** | **72** |
| 8 | 2 | 97.0% | 131.0 | 97 |
| 4 | 3 | 98.7% | 112.4 | 138 |
| 8 | 3 | 99.6% | 171.3 | 165 |

Monotone in both knobs, with a knee near L=4/r=2. At this scale brute force is also microsecond-class; the *shape* is the result.

## 5. Corpus scale: an honest loss to HNSW

Corpus: 589,933 real word-embedding rows from seven trained models (DistilBERT, GPT-2, RoBERTa, BERT, BART, DeBERTa-v3, XLM-R), 0 exact duplicates (content-hash verified). Queries: 90 perturbed probes at σ ∈ {0, 0.25, 0.5}; truth: exact top-10 by inner product (brute force 260.3 ms/query scalar Rust; 17.1 ms/query FAISS FlatIP batched, 100% recall).

| Method | recall@10 | µs/query |
|---|---|---|
| Exact (FAISS FlatIP, batched) | 100% | 17,076 |
| STAIR L=4 r=2 (best) | 58.3% | 2,260 |
| **HNSW-IP ef=16** | **74.3%** | **109** |
| HNSW-IP ef=64 | 83.7% | 317 |
| HNSW-IP ef=256 | 90.9% | 1,749 |

**HNSW wins the pure approximate-search task decisively**, at comparable build time (45 s vs 53 s) and memory. We report this as the headline of this section rather than burying it: a first-implementation sketch index should not beat a mature graph library, and our value proposition lies elsewhere — the exact tier, time semantics, and composition (Sections 6–8). Two measurement notes: (i) an HNSW configured with L2 while truth uses inner product scores 14.2% at ef=16 — the metric mismatch of Section 9.5, not engine weakness; (ii) our σ=0 timing block (726 µs at L=1) is a cold-cache artifact; later blocks are representative.

## 6. Temporal gating: when it helps

Gating (restricting candidates to ±w seconds) on two corpora:

**A. Controlled trajectory corpus** (50,000 embeddings on an Ornstein–Uhlenbeck path; ground-truth spread: mean |Δt| = 0.25 s, max 0.60 s):

| gate | recall@10 | µs/q |
|---|---|---|
| ±0.25 s | 50.0% | 71 |
| **±0.5 s** | **98.5%** | **66** (4.7× vs ungated 308 µs) |
| ±1 s | 100% | 71 |
| none | 100% | 308 |

**B. Word embeddings** (time is meaningless): the same ±2 s gate cuts recall 66% → 21%.

**Boundary condition:** measure the ground-truth temporal spread first; a gate must comfortably exceed it. Gating helps exactly when content correlates with time — and the spread measurement tells you *before* querying.

## 7. Z-key serving and time-range queries

Same trajectory corpus, L=4, r=2:

| mode | recall@10 | µs/q | cand/q | lookups/q |
|---|---|---|---|---|
| ungated (reference) | 100% | 308 | 484 | — |
| **C: Z-key gate (probe-time)** | **100%** | **68** | **17** | 25 |
| **D: time-range ranking** | **100%** | **2–4** | 17 | — |

Probe-time gating prunes 96% of candidates *before* similarity computation (time is in the key), at equal recall. Mode D answers "what is in [t−w, t+w], ranked" — a query type the pre-Z-key design and similarity-only indexes cannot express. (Two rows from this experiment's first run — a no-gate and a post-filter variant — were invalidated by an instrumentation bug in our probe mask and are excluded; we note this because exclusion criteria should be stated.)

## 8. Grid-fed attention (learned probes)

Prior retrieval-augmented approaches feed models with queries produced by the model's own representations. Our prototype tests the minimal mechanism end-to-end:

- **Memory:** 256 "facts" (random unit vectors, D=32) + 1,024 filler entries; each fact stored *perturbed* (noise 0.1/dim).
- **Probes:** each fact perturbed *independently* — bit-identical lookup impossible by construction; this is the non-circular retrieval this program previously lacked.
- **Model:** linear W_q, W_k trained by SGD (softmax cross-entropy over memory logits; final CE 0.0252).
- **Serving:** learned query → Z-window → sketch prefilter → exact rerank → attention over candidates.

| path | accuracy | dot-products/query |
|---|---|---|
| untrained control | 0.0% (chance 0.08%) | 1,280 |
| full attention (trained) | 98.6% | 1,280 |
| window-only control | 100% | 83 |
| **grid-fed (served)** | **94.9%** | **11** |

Prefilter recall: 94.9%. Served accuracy equals prefilter recall exactly — every error is a prefilter miss; rerank is never wrong when the true fact survives. **120× fewer comparisons than full attention**, with a learned query, through a 3D memory.

Honest footnotes. (i) On this task the time gate alone preserved accuracy (window-only 100%); the sketch tier's marginal contribution is comparison reduction (83 → 11), not accuracy — the two components are separable and were measured separately. (ii) At toy scale, wall-clock favors full attention (19.5 vs 5.2 µs): probe overhead dominates 32-dimensional dot products. The reduction is in comparisons — the asymptotically meaningful quantity; wall-clock crossover requires realistic dimensionality and memory sizes. (iii) This is a mechanism demonstration on synthetic data — not a language model, and not a claim about production systems.

### 8.1 Real dimensionality: 589,933 x 768 (PyTorch, MPS)

Setup: memory = the full 590K corpus (Section 5); probes = a source embedding plus per-dimension Gaussian noise at sigma in {0.25, 0.5, 1.0} (units: each dimension's own std); adapter W_q (identity init, Adam, weight decay 1e-4) trained on 8,000 probes at sigma=0.5 with softmax cross-entropy over full-memory logits (CE 1.57 at epoch 5 to 0.54 at epoch 30); prefilter = L=4 PCA-32 sketch tables, 20-bit prefix, Hamming radius 2; serving = prefilter, then exact rerank over survivors. Metric here is top-1 accuracy / source containment — NOT recall@10, so these numbers are not directly comparable to Section 5.

| sigma | full raw top-1 | full learned | prefilter recall raw | prefilter recall learned | served raw top-1 | cand/q (learned path) |
|---|---|---|---|---|---|---|
| 0.25 | 92.5% | 91.0% | 94.5% | 19.0% | 92.5% | 2,166 |
| 0.50 | 93.5% | 93.5% | 76.5% | 17.0% | 74.5% | 1,553 |
| 1.00 | 84.0% | 87.0% | 42.5% | 12.5% | 42.0% | 1,194 |

(a) Graceful degradation vs the cliff. As noise rises, the sketch tier retains 94.5% -> 76.5% -> 42.5% prefilter recall where the exact tier returns nothing at any sigma > 0 (Section 3). Served top-1 at sigma=1.0 is 42.0% using 1,194 comparisons — 494x fewer than full attention (589,933). The two-tier memory fails gradually, which is the property a memory system wants.

(b) The linear adapter has no headroom here. Full-attention accuracy moves at most +3.0 points (sigma=1.0), 0.0 (sigma=0.5), and -1.5 (sigma=0.25). Mechanism: noise is scaled per-dimension by the data's own standard deviation, so signal-to-noise ratio is uniform across dimensions; after per-dimension standardization the retrieval problem is already near linearly optimal under raw dot products, leaving a linear map almost nothing to fix.

(c) Negative result: discriminatively adapted queries are sketch-incompatible. Passing queries through the adapter collapses prefilter recall from 94.5% to 19.0% (sigma=0.25). The adapter is trained only to rank; nothing constrains the adapted query to remain near the embedding manifold the sketch directions were learned from, so its sketch bits decorrelate from stored keys' and it probes the wrong buckets. The effect is invisible at D=32 (Section 8), where the temporal window control dominated. Co-design implication: query adaptation and memory prefiltration must be jointly constrained (e.g., a proximity penalty on ||W_q q - q||, or sketch directions learned jointly with the adapter). We consider this the central open problem of grid-fed attention.

## 9. Nine pitfalls (a post-mortem)

An earlier draft by the author (never submitted; superseded by this paper) reported 12–6,876× speedups with 100% accuracy and billion-scale results. Every headline was an artifact. Each pitfall below is demonstrated with our own measured example; we believe each is common in the literature.

1. **Bounded-source "corpora."** picsum.photos serves a finite library: our "13,312 real photographs" contained **993 unique images (92.5% duplicates)**.
2. **Self-retrieval circularity.** Querying with the stored embedding itself proves dictionary lookup, not retrieval.
3. **Exact-match scoring of similarity engines.** Defining "correct" as exact-key match makes exact indexes fail by construction: on the duplicate corpus, an exact FAISS index scored "7.2% accuracy" — precisely the ~1/13.4 tie probability (7.5%) of returning a different duplicate.
4. **Check asymmetry.** The hash system was graded on bucket *membership*; FAISS on exact *argmax*. Same data, same task, different exams.
5. **Metric mismatch.** Truth by inner product, index by L2: identical HNSW configuration scores 14.2% (L2) vs 74.3% (IP) on the same queries.
6. **Cross-task speedups.** Dict lookups benchmarked against attention or similarity search and reported as "faster than attention."
7. **Language constants.** Rust hash lookups vs Python-looped FAISS: a large share of a reported "6,876×" was interpreter overhead, not algorithm.
8. **Projected baselines as measurements.** Billion-scale "FAISS cost" was a hardcoded formula, never run; "100% accuracy at 1B" was keys generated by the same formula that generated the queries.
9. **Unverifiable results.** Two reported experiments (an HNSW M-sweep; an adaptive-hash validation) had no surviving source in the repository. Results without reproducible source are not results.

Two further practices we corrected in our own code: hardcoded summary tables inside benchmark programs, and O(N²) extrapolations labeled as attention measurements.

## 10. Limitations

L3 is a toy-scale mechanism demonstration (synthetic facts, D=32, single linear head). The trajectory corpus is synthetic (by design — it isolates the gate variable). STAIR is implemented in scalar Rust without SIMD/batching; its latency comparison to FAISS is therefore conservative against us, but its memory layout is not optimized. We have not integrated grid-fed attention into a trained language model. The word-embedding corpus mixes models with different training objectives — realistic for our storage use case, but not a standard retrieval benchmark.

## 11. What we claim

- Time as a first-class key is a real, measurable capability: probe-time gating (100% recall, 4.5× faster, 28× fewer candidates) and time-range ranking (a query type similarity indexes cannot express).
- Temporal gating has a measured boundary condition: helpful iff gate > ground-truth temporal spread.
- The exact tier's σ-cliff is total (100% → 0%); two-tier composition is therefore necessary, not decorative.
- Learned probes preserve attention accuracy at toy scale (D=32, 120x fewer comparisons); at real dimensionality we measure the co-design constraint: discriminatively adapted queries collapse sketch prefiltration (94.5% to 19.0%) unless query geometry is preserved — the central open problem of grid-fed attention (Section 8.1).
- At pure approximate search, HNSW is better than our sketch tier, and we say so.

## References

[1] Charikar, *Similarity Estimation Techniques from Rounding Algorithms*, STOC 2002.
[2] Lv, Josephson, Wang, Charikar, Li, *Multi-Probe LSH*, VLDB 2007.
[3] Malkov, Yashunin, *Efficient and robust approximate nearest neighbor search using HNSW*, TPAMI 2018.
[4] Johnson, Douze, Jégou, *Billion-scale similarity search with GPUs*, IEEE Trans. Big Data 2021.
[5] Jégou, Douze, Schmid, *Product Quantization*, TPAMI 2011.
[6] Wu et al., *Memorizing Transformers*, ICLR 2022.
[7] Borgeaud et al., *Improving language models by retrieving from trillions of tokens (RETRO)*, ICML 2022.
[8] Khandelwal et al., *Nearest Neighbor Machine Translation / kNN-LM*, ICLR 2020.
[9] Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.
[10] Dosovitskiy et al., *An Image is Worth 16×16 Words (ViT)*, ICLR 2021.
