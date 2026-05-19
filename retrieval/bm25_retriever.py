"""
BM25 Retriever (Week 2)
========================
Keyword-based retrieval using rank-bm25.
Loads the corpus JSON and builds an in-memory BM25 index.
"""

import json
from pathlib import Path
from rank_bm25 import BM25Okapi

CORPUS_PATH = Path("data/corpus.json")


class BM25Retriever:
    def __init__(self, top_k: int = 20):
        self.top_k = top_k
        print("Loading BM25 index from corpus...")
        with open(CORPUS_PATH, "r") as f:
            self.corpus = json.load(f)

        tokenized = [c["text"].lower().split() for c in self.corpus]
        self.bm25 = BM25Okapi(tokenized)
        print(f"  BM25 index built over {len(self.corpus)} chunks")

    def retrieve(self, query: str) -> list[dict]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)

        # Get top_k indices sorted by score
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[: self.top_k]

        results = []
        for idx in top_indices:
            if scores[idx] == 0:
                continue
            chunk = self.corpus[idx]
            results.append({
                "text":      chunk["text"],
                "score":     float(scores[idx]),
                "source":    "bm25",
                "title":     chunk["title"],
                "arxiv_url": chunk["arxiv_url"],
                "published": chunk["published"],
                "paper_id":  chunk["paper_id"],
                "chunk_index": chunk["chunk_index"],
            })
        return results
