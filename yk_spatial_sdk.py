"""
YK-Spatial Python SDK (Phase 2)
pip install requests base64
Usage:
    from yk_spatial_sdk import YKSpatial
    yk = YKSpatial("http://localhost:3000")
    yk.quantize(weights)
    yk.index(embeddings, modality=1)
    result = yk.query(embedding, timecode=15.7, modality=1)
"""
import requests
import base64
import struct
import json

class YKSpatial:
    def __init__(self, server_url="http://localhost:3000"):
        self.url = server_url
    
    def quantize(self, weights):
        """Compress f32 weights to 4-bit packed bytes.
        
        Args:
            weights: list of floats (f32)
        Returns:
            dict: {status, original_size, packed_size, scale, packed_b64}
        """
        raw = struct.pack(f'{len(weights)}f', *weights)
        b64 = base64.b64encode(raw).decode()
        
        response = requests.post(
            f"{self.url}/quantize",
            json={"weights_b64": b64}
        )
        return response.json()
    
    def index(self, embeddings, timecodes, modality=1):
        """Index embeddings in the 3D spatial-temporal grid.
        
        Args:
            embeddings: list of 768-dim float arrays
            timecodes: list of float timecodes
            modality: 0=Text, 1=Video, 2=Audio
        Returns:
            dict: {status, items_indexed, grid_buckets}
        """
        # Pack: [num_items(u32)] [for each: timecode(f32) + 768 floats]
        data = struct.pack('I', len(embeddings))
        for i, emb in enumerate(embeddings):
            data += struct.pack('f', timecodes[i])
            data += struct.pack(f'{len(emb)}f', *emb)
        
        b64 = base64.b64encode(data).decode()
        
        response = requests.post(
            f"{self.url}/index",
            json={"embeddings_b64": b64, "modality": modality}
        )
        return response.json()
    
    def query(self, embedding, timecode, modality=1):
        """Query the 3D spatial grid (O(1) retrieval).
        
        Args:
            embedding: 768-dim float array
            timecode: float time in seconds
            modality: 0=Text, 1=Video, 2=Audio
        Returns:
            dict: {status, found, results, query_time_ns}
        """
        response = requests.post(
            f"{self.url}/query",
            json={
                "embedding": embedding,
                "timecode": timecode,
                "modality": modality
            }
        )
        return response.json()


# ==========================================
# Demo: How an ML engineer uses the SDK
# ==========================================
if __name__ == "__main__":
    print("=== YK-Spatial Python SDK Demo ===\n")
    
    yk = YKSpatial("http://localhost:3000")
    
    # 1. Quantize some weights
    print("1. Quantizing weights...")
    weights = [0.12, -0.45, 1.20, -0.03, 0.88, -1.50, 0.01, -0.99]
    result = yk.quantize(weights)
    print(f"   Original: {result['original_size']} bytes")
    print(f"   Packed:   {result['packed_size']} bytes")
    print(f"   Scale:    {result['scale']:.6f}")
    
    # 2. Index some embeddings (simulated ViT output)
    print("\n2. Indexing 3 embeddings in 3D grid...")
    embeddings = [
        [0.5, -0.3, 0.8, -0.1] + [0.0] * 764,
        [0.1, -0.8, 0.3, 0.5] + [0.0] * 764,
        [-0.2, 0.6, -0.4, 0.9] + [0.0] * 764,
    ]
    timecodes = [0.0, 1.0, 2.0]
    
    result = yk.index(embeddings, timecodes, modality=1)
    print(f"   Items indexed: {result['items_indexed']}")
    print(f"   Grid buckets:  {result['grid_buckets']}")
    
    # 3. Query: Find embedding at t=1.0
    print("\n3. Querying 3D grid for t=1.0s...")
    query_emb = [0.1, -0.8, 0.3, 0.5] + [0.0] * 764
    result = yk.query(query_emb, timecode=1.0, modality=1)
    print(f"   Found: {result['found']}")
    print(f"   Query time: {result['query_time_ns']} ns")
    if result['results']:
        print(f"   Pair ID: {result['results'][0]['pair_id']}")
        print(f"   Embedding preview: {result['results'][0]['embedding_preview']}")
    
    print("\n✅ SDK works. Any ML engineer can now use YK-Spatial from Python.")
