"""
Vector Retriever (Week 2)
==========================
Semantic similarity search using Chroma + sentence-transformers.
"""

from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR  = Path("data/chroma_db")
COLLECTION  = "arxiv_rag"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


class VectorRetriever:
    def __init__(self, top_k: int = 20):
        self.top_k = top_k
        self.model = SentenceTransformer(EMBED_MODEL)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = client.get_collection(COLLECTION)

    def retrieve(self, query: str) -> list[dict]:
        query_vec = self.model.encode(
            query, normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text":      doc,
                "score":     1 - dist,   # cosine similarity
                "source":    "vector",
                **meta,
            })
        return chunks
