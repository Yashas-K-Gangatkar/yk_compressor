# grid_attention_768.py — L3 scale-up: learned probes on 589,933 real
# embeddings at D=768, with a sketch-prefilter serving path.
#
# Task: probe = source embedding + per-dim gaussian noise (sigma * dim_std).
# The model must identify the source among ALL memory entries.
#   baseline: raw noisy query (no learning)
#   learned:  q = probe @ Wq  (Wq init = identity, trained by softmax CE
#             over full-memory logits; Adam, weight decay)
#   serving:  PCA-32 sign-sketch tables (L=4, prefix 20 bits, radius 2)
#             -> exact rerank over surviving candidates
#
# Honest scope: retrieval task on real embeddings; not a language model.
# Z-axis not tested here (word timecodes are meaningless — measured earlier);
# temporal gating composes orthogonally (demonstrated in trajectory_test.rs).
#
# RAM: ~4-5 GB. Time: training ~3-6 min (MPS), prefilter eval ~1-2 min.
# Run:  python3 grid_attention_768.py             (full corpus)
#       python3 grid_attention_768.py --n 200000  (RAM-tight machines)

import numpy as np, struct, time, argparse, torch

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=0)
ap.add_argument("--sigma-train", type=float, default=0.5)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--train-q", type=int, default=8000)
ap.add_argument("--eval-q", type=int, default=200)   # per sigma
ap.add_argument("--seed", type=int, default=1234)
args = ap.parse_args()

rng = np.random.default_rng(args.seed)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {dev}")

# ---- corpus ----
with open("yk_words.bin", "rb") as f:
    n_file = struct.unpack('I', f.read(4))[0]
n = n_file if args.n == 0 else min(args.n, n_file)
X = np.fromfile("yk_words.bin", dtype=np.float32,
                count=1 + n * 769, offset=4).reshape(n_file, 769)
E = np.ascontiguousarray(X[:n, 1:])
dim_std = E.std(axis=0).astype(np.float32)
print(f"corpus: {n} x 768 | mean dim_std {dim_std.mean():.4f}")

Et = torch.from_numpy(E).to(dev)   # fixed memory, no grad

# ---- probes ----
SIGMAS = [0.25, 0.5, 1.0]
def make_probes(count, sigma):
    src = rng.integers(0, n, size=count)
    noise = rng.normal(size=(count, 768)).astype(np.float32) * sigma * dim_std
    return (E[src] + noise).astype(np.float32), src

train_Q, train_src = make_probes(args.train_q, args.sigma_train)

# ---- learned query adapter (linear, identity init) ----
Wq = torch.eye(768, device=dev, requires_grad=True)
opt = torch.optim.Adam([Wq], lr=1e-3, weight_decay=1e-4)
lossf = torch.nn.CrossEntropyLoss()
trainQt = torch.from_numpy(train_Q).to(dev)
trainSrct = torch.from_numpy(train_src).to(dev)

print(f"training Wq: {args.epochs} epochs, {args.train_q} probes, "
      f"sigma={args.sigma_train}, batch={args.batch}")
t0 = time.time()
for ep in range(args.epochs):
    perm = torch.randperm(args.train_q, device=dev)
    tot = 0.0
    for i in range(0, args.train_q, args.batch):
        idx = perm[i:i + args.batch]
        logits = (trainQt[idx] @ Wq) @ Et.T
        loss = lossf(logits, trainSrct[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(idx)
    if (ep + 1) % 5 == 0:
        print(f"  epoch {ep+1}: CE {tot/args.train_q:.4f}")
print(f"trained in {time.time()-t0:.1f}s")
Wq_np = Wq.detach().cpu().numpy().astype(np.float32)

# ---- full-attention eval (chunked) ----
def eval_full(Qnp, src, adapted, chunk=64):
    with torch.no_grad():
        Qt = torch.from_numpy(Qnp).to(dev)
        if adapted: Qt = Qt @ Wq
        top1 = 0; r10 = 0
        for i in range(0, len(Qt), chunk):
            logits = Qt[i:i+chunk] @ Et.T
            top = logits.topk(10, dim=1).indices.cpu().numpy()
            src_c = src[i:i+chunk]
            top1 += (top[:, 0] == src_c).sum()
            r10 += (top == src_c[:, None]).any(axis=1).sum()
    m = len(Qnp)
    return 100.0 * top1 / m, 100.0 * r10 / m

# ---- sketch prefilter: L=4 PCA-32 tables, prefix 20 bits, radius 2 ----
P_BITS, RADIUS, L_TABLES, DIRS = 20, 2, 4, 32
masks = [0] + [m for k in (1, 2) for m in range(1 << P_BITS)
               if m.bit_count() == k]
print(f"prefilter: L={L_TABLES} tables, {DIRS} PCA dirs, "
      f"prefix={P_BITS} bits, radius={RADIUS} ({len(masks)} probes/table)")

def build_table(seed):
    r = np.random.default_rng(seed)
    idx = r.choice(n, size=min(20000, n), replace=False)
    S = E[idx]; mu = S.mean(0); Xc = S - mu
    C = (Xc.T @ Xc) / len(idx)
    w, v = np.linalg.eigh(C)              # ascending eigvals
    dirs = v[:, -DIRS:].T                 # top-32 PCs, shape (32, 768)
    code = ((E - mu) @ dirs.T) > 0        # (n, 32)
    prefix = np.zeros(n, dtype=np.int64)
    for b in range(P_BITS):
        prefix = (prefix << 1) | code[:, b]
    order = np.argsort(prefix, kind="stable")
    return mu, dirs, order, prefix[order]

t0 = time.time()
tables = [build_table(1000 + t) for t in range(L_TABLES)]
print(f"prefilter built in {time.time()-t0:.1f}s")

def sketch_prefix(Qnp, mu, dirs):
    code = ((Qnp - mu) @ dirs.T) > 0
    p = np.zeros(len(Qnp), dtype=np.int64)
    for b in range(P_BITS):
        p = (p << 1) | code[:, b]
    return p

def prefilter(Qnp):
    cands = [set() for _ in range(len(Qnp))]
    for mu, dirs, order, sp in tables:
        qp = sketch_prefix(Qnp, mu, dirs)
        for qi in range(len(Qnp)):
            s = cands[qi]
            for mk in masks:
                key = int(qp[qi]) ^ mk
                lo = np.searchsorted(sp, key, side="left")
                hi = np.searchsorted(sp, key, side="right")
                if hi > lo:
                    s.update(order[lo:hi].tolist())
    return cands

def eval_served(Qnp, src, cands, k=10):
    top1 = 0; r10 = 0; pre = 0; cc = 0
    for qi, s in enumerate(cands):
        cc += len(s)
        if src[qi] in s: pre += 1
        if s:
            ids = np.fromiter(s, dtype=np.int64)
            scores = E[ids] @ Qnp[qi]
            top = ids[np.argsort(-scores)[:k]]
            top1 += (top[0] == src[qi])
            r10 += (top == src[qi]).any()
    m = len(Qnp)
    return (100.0 * top1 / m, 100.0 * r10 / m,
            100.0 * pre / m, cc / m)

# ---- run everything ----
print("\n" + "=" * 100)
print(f"{'sigma':>5} | {'full raw':>8} | {'full learned':>12} | "
      f"{'pre-rec raw':>11} | {'pre-rec lrnd':>12} | "
      f"{'served raw':>10} | {'served lrnd':>11} | {'cand/q':>7}")
print("-" * 100)
for sigma in SIGMAS:
    Qnp, src = make_probes(args.eval_q, sigma)
    fr = eval_full(Qnp, src, adapted=False)
    fl = eval_full(Qnp, src, adapted=True)
    Ql = (Qnp @ Wq_np).astype(np.float32)
    c_raw = prefilter(Qnp)
    c_lrn = prefilter(Ql)
    sr = eval_served(Qnp, src, c_raw)
    sl = eval_served(Ql, src, c_lrn)
    print(f"{sigma:>5.2f} | {fr[0]:>7.1f}% | {fl[0]:>11.1f}% | "
          f"{sr[2]:>10.1f}% | {sl[2]:>11.1f}% | "
          f"{sr[0]:>9.1f}% | {sl[0]:>10.1f}% | {sl[3]:>7.0f}")

print("\ncomparisons/query: full attention = {:,}; served = cand/q above "
      "({:.0f}x reduction at last sigma)".format(n, n / sl[3] if sl[3] else 0))
print("reading:")
print("- full learned vs full raw: does Wq beat the identity init? (main L3 claim at D=768)")
print("- pre-rec learned vs raw: does the learned query ALSO sketch better? (compounding)")
print("- served ~= pre-rec x full: accuracy is prefilter-limited (same as Rust prototype)")
print("- if pre-rec is the ceiling, raise L_TABLES or RADIUS (trade-off already mapped in stair_sweep)")
