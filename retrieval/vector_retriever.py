"""
Vector Retriever — Qdrant Cloud
=================================
Semantic similarity search using Qdrant Cloud + sentence-transformers.
Replaces the local Chroma-based retriever.

Usage:
    from retrieval.vector_retriever import VectorRetriever
    r = VectorRetriever(top_k=20)
    results = r.retrieve("What are failure modes of naive RAG?")
"""

from sentence_transformers import SentenceTransformer
from retrieval.qdrant_client_factory import get_client, COLLECTION_NAME

EMBED_MODEL = "BAAI/bge-small-en-v1.5"


class VectorRetriever:
    def __init__(self, top_k: int = 20):
        self.top_k  = top_k
        self.model  = SentenceTransformer(EMBED_MODEL, device="cpu")
        self.client = get_client()

    def retrieve(self, query: str) -> list[dict]:
        # Embed the query
        query_vec = self.model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        # Search Qdrant Cloud
        results = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            limit=self.top_k,
            with_payload=True,
        )

        # Format results to match the same schema as before
        chunks = []
        for hit in results:
            payload = hit.payload
            chunks.append({
                "text":        payload["text"],
                "score":       hit.score,
                "source":      "vector",
                "title":       payload["title"],
                "arxiv_url":   payload["arxiv_url"],
                "published":   payload["published"],
                "paper_id":    payload["paper_id"],
                "chunk_index": payload["chunk_index"],
            })

        return chunks
