"""
Embed Corpus → Qdrant Cloud
=============================
Reads data/corpus.json, generates embeddings with sentence-transformers,
and uploads everything to your Qdrant Cloud cluster.

Run once to populate the cloud vector DB:
    python ingestion/embed_corpus.py

Requirements:
    pip install qdrant-client sentence-transformers tqdm
"""

import json
import sys
from pathlib import Path

# Allow `python ingestion/embed_corpus.py` without PYTHONPATH=.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct
)
from sentence_transformers import SentenceTransformer
from retrieval.qdrant_client_factory import get_client, COLLECTION_NAME, VECTOR_SIZE

# ── Config ────────────────────────────────────────────────────────────────────

CORPUS_PATH = Path("data/corpus.json")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
BATCH_SIZE  = 32
DEVICE      = "cpu"  # avoids MPS OOM on Mac during encode
UPSERT_BATCH = 100   # Qdrant recommends smaller batches than Chroma

# ── Load corpus ───────────────────────────────────────────────────────────────

print(f"Loading corpus from {CORPUS_PATH}")
with open(CORPUS_PATH, "r") as f:
    corpus = json.load(f)
print(f"  {len(corpus)} chunks loaded")

# ── Embed ─────────────────────────────────────────────────────────────────────

print(f"\nLoading embedding model: {EMBED_MODEL} (device={DEVICE})")
model = SentenceTransformer(EMBED_MODEL, device=DEVICE)

texts = [c["text"] for c in corpus]
print(f"Embedding {len(texts)} chunks (batch size {BATCH_SIZE})...")

embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,
)

# ── Connect to Qdrant Cloud ───────────────────────────────────────────────────

print(f"\nConnecting to Qdrant Cloud...")
client = get_client()

# Recreate collection (safe to re-run)
try:
    client.delete_collection(COLLECTION_NAME)
    print(f"  Deleted existing collection '{COLLECTION_NAME}'")
except Exception:
    pass

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=VECTOR_SIZE,
        distance=Distance.COSINE,
    ),
)
print(f"  Created collection '{COLLECTION_NAME}' (cosine, {VECTOR_SIZE}d)")

# ── Upload vectors ────────────────────────────────────────────────────────────

print(f"\nUploading {len(corpus)} vectors to Qdrant Cloud...")

for i in tqdm(range(0, len(corpus), UPSERT_BATCH), desc="Uploading"):
    batch      = corpus[i : i + UPSERT_BATCH]
    batch_vecs = embeddings[i : i + UPSERT_BATCH]

    points = [
        PointStruct(
            id=idx,   # Qdrant needs integer IDs
            vector=vec.tolist(),
            payload={
                "chunk_id":    c["chunk_id"],
                "text":        c["text"],
                "title":       c["title"],
                "authors":     c["authors"],
                "published":   c["published"],
                "arxiv_url":   c["arxiv_url"],
                "paper_id":    c["paper_id"],
                "chunk_index": c["chunk_index"],
            }
        )
        for idx, (c, vec) in enumerate(zip(batch, batch_vecs), start=i)
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)

# ── Verify ────────────────────────────────────────────────────────────────────

info = client.get_collection(COLLECTION_NAME)
print(f"\n Done!")
print(f"  Vectors in Qdrant Cloud : {info.points_count}")
print(f"  Collection              : {COLLECTION_NAME}")
print(f"  Next step: test with retrieval/vector_retriever.py")
