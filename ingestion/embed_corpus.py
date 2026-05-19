"""
Embed Corpus (Week 1 — final step)
====================================
Reads data/corpus.json, generates embeddings with sentence-transformers,
and stores everything in a local Chroma vector database.

Usage:
    python ingestion/embed_corpus.py

Requirements:
    pip install chromadb sentence-transformers tqdm
"""

import json
from pathlib import Path
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────

CORPUS_PATH = Path("data/corpus.json")
CHROMA_DIR  = Path("data/chroma_db")
COLLECTION  = "arxiv_rag"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
BATCH_SIZE  = 32   # lower batch avoids MPS/RAM OOM on Mac
DEVICE      = "cpu"  # MPS often OOMs during encode; CPU is fine for ~650 chunks

# ── Load ──────────────────────────────────────────────────────────────────────

print(f"Loading corpus from {CORPUS_PATH}")
with open(CORPUS_PATH, "r") as f:
    corpus = json.load(f)
print(f"  {len(corpus)} chunks loaded")

# ── Embed ─────────────────────────────────────────────────────────────────────

print(f"\nLoading embedding model: {EMBED_MODEL} (device={DEVICE})")
model = SentenceTransformer(EMBED_MODEL, device=DEVICE)

texts = [c["text"] for c in corpus]
print(f"Embedding {len(texts)} chunks...")

embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,
)

# ── Store ─────────────────────────────────────────────────────────────────────

print(f"\nStoring in Chroma at {CHROMA_DIR}")
client = chromadb.PersistentClient(path=str(CHROMA_DIR))

try:
    client.delete_collection(COLLECTION)
except Exception:
    pass

collection = client.create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "cosine"},
)

UPSERT_BATCH = 500
for i in tqdm(range(0, len(corpus), UPSERT_BATCH), desc="Upserting"):
    batch = corpus[i : i + UPSERT_BATCH]
    collection.upsert(
        ids        = [c["chunk_id"] for c in batch],
        documents  = [c["text"] for c in batch],
        embeddings = embeddings[i : i + UPSERT_BATCH].tolist(),
        metadatas  = [
            {
                "title":       c["title"],
                "arxiv_url":   c["arxiv_url"],
                "published":   c["published"],
                "paper_id":    c["paper_id"],
                "chunk_index": c["chunk_index"],
            }
            for c in batch
        ],
    )

print(f"\n Done! Collection '{COLLECTION}' has {collection.count()} vectors")
print("   Next step: build retrieval/hybrid_retriever.py")
