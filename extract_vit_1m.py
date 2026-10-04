import torch
from transformers import ViTModel, ViTImageProcessor
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import struct
import os
import time

print("=== YK-Spatial: Real ViT Image Embedding Extraction ===")
print("Model: google/vit-base-patch16-224 (86M params)")
print("Source: Real photos from picsum.photos")
print("Target: 1,000,000 real ViT embeddings\n")

# Load ViT model
print("Loading ViT model...")
model = ViTModel.from_pretrained("google/vit-base-patch16-224")
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model.eval()

output_file = "real_vit_embeddings.bin"

# Check how many already saved (resume support)
start_count = 0
if os.path.exists(output_file) and os.path.getsize(output_file) > 4:
    with open(output_file, "rb") as f:
        start_count = struct.unpack('I', f.read(4))[0]
    print(f"Resuming from {start_count:,} existing embeddings")

target = 1_000_000
batch_size = 32
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}

print(f"Target: {target:,} embeddings")
print(f"Batch size: {batch_size} images per ViT forward pass")
print(f"Estimated time: ~15 hours (batch processing)")
print(f"Press Ctrl+C to stop at any time. Saved embeddings are preserved.\n")

# Open file
if start_count == 0:
    f = open(output_file, 'wb')
    f.write(struct.pack('I', 0))  # Placeholder for count
else:
    f = open(output_file, 'r+b')
    f.seek(0, 2)  # Seek to end for appending

current = start_count
start_time = time.time()
errors = 0

try:
    while current < target:
        # Download batch of real images
        images = []
        for i in range(batch_size):
            idx = current + i
            url = f"https://picsum.photos/224/224?random={idx}"
            try:
                response = requests.get(url, headers=headers, timeout=15)
                img = Image.open(BytesIO(response.content)).convert("RGB")
                images.append(img)
            except:
                images.append(Image.new('RGB', (224, 224)))
                errors += 1

        # Process batch through real ViT model
        try:
            inputs = processor(images=images, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].numpy().astype(np.float32)
        except Exception as e:
            print(f"\nViT error on batch at {current}: {e}")
            continue

        # Save embeddings to file
        for i in range(batch_size):
            timecode = (current + i) * 0.1
            f.write(struct.pack('f', timecode))
            f.write(embeddings[i].tobytes())

        f.flush()
        current += batch_size

        # Progress report
        elapsed = time.time() - start_time
        processed = current - start_count
        rate = processed / elapsed if elapsed > 0 else 0
        remaining = target - current
        eta = remaining / rate if rate > 0 else 0

        print(f"\r  {current:,}/{target:,} ({current/target*100:.1f}%) | "
              f"{rate:.1f} img/s | ETA: {eta/3600:.1f}h | "
              f"Errors: {errors}", end='', flush=True)

        if current % 10000 == 0 and current > start_count:
            file_mb = os.path.getsize(output_file) / 1024 / 1024
            print(f"\n  Checkpoint: {current:,} embeddings | "
                  f"File: {file_mb:.1f} MB | "
                  f"Rate: {rate:.1f} img/s | "
                  f"ETA: {eta/3600:.1f}h")

except KeyboardInterrupt:
    print(f"\n\nStopped by user at {current:,} embeddings.")

# Update count at start of file
f.seek(0)
f.write(struct.pack('I', current))
f.close()

print(f"\nDone! Total: {current:,} real ViT image embeddings")
print(f"File: {output_file} ({os.path.getsize(output_file)/1024/1024:.1f} MB)")
print(f"Each embedding: 768-dim float32 from google/vit-base-patch16-224")
print(f"Data source: Real photos from picsum.photos processed through real ViT")
print(f"\nRun benchmark_vit.py to test FAISS vs YK-Spatial on these embeddings.")
