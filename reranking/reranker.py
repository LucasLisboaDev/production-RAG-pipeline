"""
Cross-Encoder Reranker (Week 3)
=================================
Takes the top-K hybrid results and re-scores them with a
cross-encoder model for maximum relevance precision.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Fast (6-layer MiniLM), free, runs on CPU
  - Trained on MS MARCO passage ranking

Usage:
    from reranking.reranker import Reranker
    reranker = Reranker()
    reranked = reranker.rerank(query, chunks, top_k=5)
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self):
        print(f"Loading reranker: {RERANKER_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            RERANKER_MODEL
        )
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def rerank(self, query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
        if not chunks:
            return []

        pairs = [(query, c["text"]) for c in chunks]

        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze(-1).cpu().tolist()

        # Attach scores and sort
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = score

        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
