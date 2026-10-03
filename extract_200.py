import torch
from transformers import ViTModel, ViTImageProcessor
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import struct
import os

print("=== YK-Spatial: 200 Real Image ViT Extraction ===")
print("Loading ViT model...")
model = ViTModel.from_pretrained("google/vit-base-patch16-224")
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model.eval()

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
num_images = 200

print(f"Downloading and processing {num_images} real images...\n")

with open("real_200.bin", "wb") as f:
    f.write(struct.pack('I', num_images))
    for i in range(num_images):
        url = f"https://picsum.photos/800/600?random={i}"
        try:
            response = requests.get(url, headers=headers, timeout=15)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            inputs = processor(images=img, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            emb = outputs.last_hidden_state[0, 0].numpy().astype(np.float32)
        except Exception as e:
            print(f"  Image {i} failed: {e}")
            emb = np.zeros(768, dtype=np.float32)
        
        timecode = float(i) * 0.1  # 0.0s to 20.0s
        f.write(struct.pack('f', timecode))
        f.write(emb.tobytes())
        
        if (i+1) % 20 == 0:
            print(f"  Processed {i+1}/{num_images} images...")

file_size = os.path.getsize("real_200.bin")
print(f"\nDone! Saved {num_images} real ViT embeddings to real_200.bin")
print(f"File size: {file_size} bytes ({file_size / 1024:.2f} KB)")
