import numpy as np, struct, hashlib, gc
from transformers import AutoModel

MODELS = [
    "distilbert/distilbert-base-uncased",
    "gpt2",
    "FacebookAI/roberta-base",
    "google-bert/bert-base-uncased",
    "facebook/bart-base",
    "microsoft/deberta-v3-base",
    "xlm-roberta-base",
]

count = 0
skipped = []
digests = set()
dupes = 0

with open("yk_words.bin", "wb") as f:
    f.write(struct.pack('I', 0))
    for name in MODELS:
        print(f"Loading {name} ...")
        model = AutoModel.from_pretrained(name)
        model.eval()
        W = model.get_input_embeddings().weight.detach().numpy().astype(np.float32)
        n, d = W.shape
        if d != 768:
            print(f"  dim={d}, skipping")
            skipped.append(name)
            del model, W
            gc.collect()
            continue
        for i in range(n):
            row = W[i]
            f.write(struct.pack('f', count * 0.1))
            f.write(row.tobytes())
            dg = hashlib.sha1(row.tobytes()).digest()
            if dg in digests:
                dupes += 1
            else:
                digests.add(dg)
            count += 1
        print(f"  {name}: {n} rows, cumulative {count}")
        del model, W
        gc.collect()

with open("yk_words.bin", "r+b") as f:
    f.seek(0)
    f.write(struct.pack('I', count))

print()
print(f"Saved {count} embeddings to yk_words.bin")
if skipped:
    print(f"Skipped (dim!=768): {skipped}")
print(f"Exact duplicate rows: {dupes} ({100.0*dupes/max(count,1):.2f}%)")
print(f"Unique rows: {count - dupes}")
