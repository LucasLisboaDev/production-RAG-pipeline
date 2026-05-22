"""
BM25 Retriever
===============
Keyword-based retrieval using rank-bm25.
Falls back gracefully if corpus.json is missing or empty
(e.g. in cloud deployments where corpus is not committed).
"""

import json
from pathlib import Path
from rank_bm25 import BM25Okapi

CORPUS_PATH = Path("data/corpus.json")


class BM25Retriever:
    def __init__(self, top_k: int = 20):
        self.top_k   = top_k
        self.corpus  = []
        self.bm25    = None
        self._load()

    def _load(self):
        if not CORPUS_PATH.exists():
            print("  BM25: corpus.json not found — running in vector-only mode")
            return

        try:
            with open(CORPUS_PATH, "r") as f:
                content = f.read().strip()
            if not content:
                print("  BM25: corpus.json is empty — running in vector-only mode")
                return
            self.corpus = json.loads(content)
            tokenized   = [c["text"].lower().split() for c in self.corpus]
            self.bm25   = BM25Okapi(tokenized)
            print(f"  BM25 index built over {len(self.corpus)} chunks")
        except Exception as e:
            print(f"  BM25: failed to load corpus ({e}) — running in vector-only mode")

    def retrieve(self, query: str) -> list[dict]:
        # If corpus unavailable, return empty list — hybrid retriever handles this
        if self.bm25 is None or not self.corpus:
            return []

        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)

        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[: self.top_k]

        results = []
        for idx in top_indices:
            if scores[idx] == 0:
                continue
            chunk = self.corpus[idx]
            results.append({
                "text":        chunk["text"],
                "score":       float(scores[idx]),
                "source":      "bm25",
                "title":       chunk["title"],
                "arxiv_url":   chunk["arxiv_url"],
                "published":   chunk["published"],
                "paper_id":    chunk["paper_id"],
                "chunk_index": chunk["chunk_index"],
            })
        return results