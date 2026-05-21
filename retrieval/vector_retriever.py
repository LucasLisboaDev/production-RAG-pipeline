"""
Vector Retriever — Qdrant Cloud
=================================
Semantic similarity search using Qdrant Cloud + sentence-transformers.
"""

from sentence_transformers import SentenceTransformer
from retrieval.qdrant_client_factory import get_client, COLLECTION_NAME

EMBED_MODEL = "BAAI/bge-small-en-v1.5"


class VectorRetriever:
    def __init__(self, top_k: int = 20):
        self.top_k  = top_k
        self.model  = SentenceTransformer(EMBED_MODEL)
        self.client = get_client()

    def retrieve(self, query: str) -> list[dict]:
        query_vec = self.model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=self.top_k,
            with_payload=True,
        ).points

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