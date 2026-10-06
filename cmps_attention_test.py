import numpy as np
import struct
import json
import time
import math

print("=== CMPS Attention vs Dot Product Attention ===")
print("THE QUESTION: Can 3D grid proximity replace dot products?\n")

# ==========================================
# 1. Load Real Data
# ==========================================
print("Loading real BERT word embeddings...")
with open("bert_model.safetensors", "rb") as f:
    header_len = struct.unpack('<Q', f.read(8))[0]
    header = json.loads(f.read(header_len))
    data_start = 8 + header_len
    
    # Load Q, K, V weights from Layer 0
    tensors = {}
    for name in ["distilbert.transformer.layer.0.attention.q_lin.weight",
                 "distilbert.transformer.layer.0.attention.q_lin.bias",
                 "distilbert.transformer.layer.0.attention.k_lin.weight",
                 "distilbert.transformer.layer.0.attention.k_lin.bias",
                 "distilbert.transformer.layer.0.attention.v_lin.weight",
                 "distilbert.transformer.layer.0.attention.v_lin.bias",
                 "distilbert.transformer.layer.0.attention.out_lin.weight",
                 "distilbert.transformer.layer.0.attention.out_lin.bias"]:
        info = header[name]
        start = info["data_offsets"][0] + data_start
        end = info["data_offsets"][1] + data_start
        f.seek(start)
        raw = f.read(end - start)
        shape = info["shape"]
        tensors[name] = np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()

    # Load word embeddings
    info = header["distilbert.embeddings.word_embeddings.weight"]
    start = info["data_offsets"][0] + data_start
    end = info["data_offsets"][1] + data_start
    f.seek(start)
    raw = f.read(end - start)
    word_embeddings = np.frombuffer(raw, dtype=np.float32).reshape(info["shape"]).copy()

dim = 768
print(f"Loaded: {word_embeddings.shape[0]} word embeddings, dim={dim}")
print(f"Loaded: real DistilBERT Layer 0 attention weights (Q, K, V, Out)")

# ==========================================
# 2. Create Input (10 tokens from real embeddings)
# ==========================================
seq_len = 10
input_indices = [100, 500, 1000, 1500, 2000, 2500, 3000, 5000, 8000, 12000]
input_embeddings = word_embeddings[input_indices]  # [10, 768]
print(f"\nInput: {seq_len} tokens from real BERT word embeddings")

# ==========================================
# 3. STANDARD ATTENTION (Dot Product)
# ==========================================
print("\n--- Standard Attention (Dot Product) ---")

Wq = tensors["distilbert.transformer.layer.0.attention.q_lin.weight"]  # [768, 768]
bq = tensors["distilbert.transformer.layer.0.attention.q_lin.bias"]
Wk = tensors["distilbert.transformer.layer.0.attention.k_lin.weight"]
bk = tensors["distilbert.transformer.layer.0.attention.k_lin.bias"]
Wv = tensors["distilbert.transformer.layer.0.attention.v_lin.weight"]
bv = tensors["distilbert.transformer.layer.0.attention.v_lin.bias"]
Wo = tensors["distilbert.transformer.layer.0.attention.out_lin.weight"]
bo = tensors["distilbert.transformer.layer.0.attention.out_lin.bias"]

def linear(x, W, b):
    return x @ W.T + b

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis, keepdims=True)

# Standard attention
t0 = time.time()
Q = linear(input_embeddings, Wq, bq)  # [10, 768]
K = linear(input_embeddings, Wk, bk)  # [10, 768]
V = linear(input_embeddings, Wv, bv)  # [10, 768]

# Scaled dot product attention
scale = 1.0 / math.sqrt(dim)
scores = (Q @ K.T) * scale  # [10, 10]
weights = softmax(scores, axis=-1)  # [10, 10]
attn_output = weights @ V  # [10, 768]

# Output projection
standard_output = linear(attn_output, Wo, bo)  # [10, 768]
standard_time = time.time() - t0

print(f"  Q, K, V computed from real weights")
print(f"  Attention scores: [10x10] dot product")
print(f"  Time: {standard_time*1e6:.1f} us")
print(f"  Output shape: {standard_output.shape}")
print(f"  Output[0][:5]: {standard_output[0][:5]}")

# ==========================================
# 4. CMPS ATTENTION (3D Grid Proximity) - NEW RULE
# ==========================================
print("\n--- CMPS Attention (3D Grid Proximity) ---")
print("NEW RULE: No Q, K, V projections needed.")
print("         Hash raw embeddings to 3D coordinates.")
print("         Attention score = 3D proximity, not dot product.")

def yk_hash(emb):
    h = 0
    for i in range(8):
        scaled = int(emb[i] * 10000)
        h = (h * 31 + scaled) & 0xFFFFFFFF
    return h

# NEW RULE: Hash all input embeddings to 3D coordinates
hashes = [yk_hash(input_embeddings[i]) for i in range(seq_len)]
timecodes = list(range(seq_len))  # Token positions as "time"
modalities = [1] * seq_len  # All same modality

# CMPS parameters
tau = 2.0   # Temporal decay (2 tokens = nearby tokens get high weight)
sigma = 2**20  # Semantic decay (hash space radius)

# NEW RULE: CMPS attention = modality_match * exp(-dz/tau) * exp(-dx/sigma)
t0 = time.time()
cmps_scores = np.zeros((seq_len, seq_len), dtype=np.float32)

for i in range(seq_len):
    for j in range(seq_len):
        # Modality match (always 1 since same modality)
        mod_match = 1.0 if modalities[i] == modalities[j] else 0.0
        
        # Temporal distance (token position)
        dz = abs(timecodes[i] - timecodes[j])
        
        # Semantic distance (hash space)
        dx = min(abs(hashes[i] - hashes[j]), 2**32 - abs(hashes[i] - hashes[j]))
        
        # CMPS score
        cmps_scores[i][j] = mod_match * math.exp(-dz / tau) * math.exp(-dx / sigma)

cmps_weights = softmax(cmps_scores, axis=-1)

# CMPS output = weighted sum of RAW embeddings (not projected V)
cmps_output = cmps_weights @ input_embeddings  # [10, 768]

# Apply output projection (same as standard)
cmps_final = linear(cmps_output, Wo, bo)
cmps_time = time.time() - t0

print(f"  Hashes computed for {seq_len} tokens")
print(f"  CMPS scores: [10x10] 3D proximity")
print(f"  Time: {cmps_time*1e6:.1f} us")
print(f"  Output shape: {cmps_final.shape}")
print(f"  Output[0][:5]: {cmps_final[0][:5]}")

# ==========================================
# 5. QUALITY COMPARISON (THE KEY TEST)
# ==========================================
print("\n" + "=" * 60)
print("QUALITY COMPARISON: Standard vs CMPS")
print("=" * 60)

# Cosine similarity between standard and CMPS output
dot_product = np.sum(standard_output * cmps_final, axis=1)
mag_std = np.linalg.norm(standard_output, axis=1)
mag_cmps = np.linalg.norm(cmps_final, axis=1)
cosine_sim = dot_product / (mag_std * mag_cmps + 1e-8)

print(f"\n| Token | Standard[0:5] | CMPS[0:5] | Cosine Sim |")
print(f"|-------|---------------|-----------|------------|")
for i in range(seq_len):
    print(f"| {i:5d} | {standard_output[i][0]:.4f} | {cmps_final[i][0]:.4f} | {cosine_sim[i]:.4f} |")

print(f"\n--- Metrics ---")
print(f"Mean cosine similarity: {np.mean(cosine_sim):.4f}")
print(f"Min cosine similarity:  {np.min(cosine_sim):.4f}")
print(f"Max cosine similarity:  {np.max(cosine_sim):.4f}")

# MSE
mse = np.mean((standard_output - cmps_final)**2)
max_err = np.max(np.abs(standard_output - cmps_final))
print(f"MSE: {mse:.6f}")
print(f"Max error: {max_err:.6f}")

# ==========================================
# 6. SPEED COMPARISON
# ==========================================
print(f"\n--- Speed ---")
print(f"Standard attention: {standard_time*1e6:.1f} us")
print(f"CMPS attention:     {cmps_time*1e6:.1f} us")
print(f"CMPS speedup:       {standard_time/cmps_time:.1f}x")

# ==========================================
# 7. VERDICT
# ==========================================
print(f"\n{'=' * 60}")
print("VERDICT: CAN CMPS REPLACE DOT PRODUCTS?")
print(f"{'=' * 60}")

mean_sim = np.mean(cosine_sim)
min_sim = np.min(cosine_sim)

if mean_sim > 0.95 and min_sim > 0.90:
    print(f"\n✅ YES! CMPS can replace dot products.")
    print(f"   Mean similarity: {mean_sim:.4f} (> 95%)")
    print(f"   Min similarity:  {min_sim:.4f} (> 90%)")
    print(f"   CMPS is {standard_time/cmps_time:.1f}x faster")
    print(f"   This is a genuine contribution worth publishing.")
elif mean_sim > 0.85:
    print(f"\n⚠️ PARTIALLY. CMPS is close but not perfect.")
    print(f"   Mean similarity: {mean_sim:.4f} (need > 95%)")
    print(f"   Min similarity:  {min_sim:.4f} (need > 90%)")
    print(f"   CMPS produces SIMILAR but not IDENTICAL attention.")
    print(f"   Might work for some applications but not all.")
else:
    print(f"\n❌ NO. CMPS cannot replace dot products.")
    print(f"   Mean similarity: {mean_sim:.4f} (need > 95%)")
    print(f"   Min similarity:  {min_sim:.4f} (need > 90%)")
    print(f"   CMPS loses too much information.")
    print(f"   Dot products capture nuanced similarity that")
    print(f"   hash proximity cannot represent.")

print(f"\n{'=' * 60}")
print("WHY THIS MATTERS")
print(f"{'=' * 60}")
print(f"""
Standard attention: Q·K^T / √d → O(d) per pair, O(N²d) total
CMPS attention:      exp(-Δz/τ) × exp(-Δx/σ) → O(1) per pair, O(N²) total

If CMPS works (similarity > 95%):
  → Remove Q, K projections (saves 2 × d² parameters)
  → Attention cost: O(N²) instead of O(N²d)
  → At d=768: 768x fewer operations
  → At d=4096: 4096x fewer operations

If CMPS fails (similarity < 95%):
  → Hash proximity is too coarse for attention
  → Dot products capture nuances that hashing loses
  → CMPS is not viable for attention replacement

This test determines whether CMPS is science or engineering.
""")

# ==========================================
# 8. DETAILED ANALYSIS
# ==========================================
print(f"{'=' * 60}")
print("DETAILED ANALYSIS")
print(f"{'=' * 60}")

print(f"\nStandard attention scores (Q·K^T/√d):")
for i in range(3):
    row = " ".join([f"{scores[i][j]:.3f}" for j in range(5)])
    print(f"  Token {i}: [{row} ...]")

print(f"\nCMPS scores (3D proximity):")
for i in range(3):
    row = " ".join([f"{cmps_scores[i][j]:.3f}" for j in range(5)])
    print(f"  Token {i}: [{row} ...]")

print(f"\nStandard attention weights (after softmax):")
for i in range(3):
    row = " ".join([f"{weights[i][j]:.3f}" for j in range(5)])
    print(f"  Token {i}: [{row} ...]")

print(f"\nCMPS attention weights (after softmax):")
for i in range(3):
    row = " ".join([f"{cmps_weights[i][j]:.3f}" for j in range(5)])
    print(f"  Token {i}: [{row} ...]")

# Analyze WHERE CMPS fails
print(f"\nAnalysis of failure modes:")
for i in range(seq_len):
    diff = np.abs(weights[i] - cmps_weights[i])
    max_diff = np.max(diff)
    argmax_diff = np.argmax(diff)
    print(f"  Token {i}: max weight diff = {max_diff:.4f} (at position {argmax_diff})")
