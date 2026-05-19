"""
Hybrid Retriever — RRF Fusion (Week 2)
========================================
Combines BM25 + vector results using Reciprocal Rank Fusion (RRF).
RRF score = sum of 1 / (k + rank) across both result lists.

Usage:
    from retrieval.hybrid_retriever import HybridRetriever
    retriever = HybridRetriever()
    chunks = retriever.retrieve("What are failure modes of naive RAG?")
"""

from retrieval.bm25_retriever import BM25Retriever
from retrieval.vector_retriever import VectorRetriever

RRF_K = 60   # standard constant; higher = less weight on top ranks


class HybridRetriever:
    def __init__(self, top_k: int = 20, bm25_k: int = 30, vector_k: int = 30):
        self.top_k = top_k
        self.bm25   = BM25Retriever(top_k=bm25_k)
        self.vector = VectorRetriever(top_k=vector_k)

    def retrieve(self, query: str) -> list[dict]:
        bm25_results   = self.bm25.retrieve(query)
        vector_results = self.vector.retrieve(query)

        # Build a map: chunk_id → chunk dict
        chunks_by_id: dict[str, dict] = {}

        for rank, chunk in enumerate(bm25_results):
            cid = f"{chunk['paper_id']}_chunk_{chunk['chunk_index']:04d}"
            if cid not in chunks_by_id:
                chunks_by_id[cid] = {**chunk, "rrf_score": 0.0}
            chunks_by_id[cid]["rrf_score"] += 1 / (RRF_K + rank + 1)

        for rank, chunk in enumerate(vector_results):
            cid = f"{chunk['paper_id']}_chunk_{chunk['chunk_index']:04d}"
            if cid not in chunks_by_id:
                chunks_by_id[cid] = {**chunk, "rrf_score": 0.0}
            chunks_by_id[cid]["rrf_score"] += 1 / (RRF_K + rank + 1)

        # Sort by RRF score and return top_k
        ranked = sorted(
            chunks_by_id.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )
        return ranked[: self.top_k]
