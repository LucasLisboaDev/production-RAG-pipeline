"""
FastAPI Application
====================
Production RAG pipeline exposed as a REST API.

Run locally:
    uvicorn api.main:app --reload

Endpoints:
    POST /query   — Ask a question, get a cited answer
    GET  /health  — Health check
"""

from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from retrieval.hybrid_retriever import HybridRetriever
from reranking.reranker import Reranker
from generation.prompt_builder import build_prompt, format_citations
from generation.llm_client import call_llm

app = FastAPI(
    title="ArXiv RAG API",
    description="Ask questions about AI research papers with cited answers.",
    version="1.0.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://*.vercel.app",
        "*",  # tighten this once you have your Vercel URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy model loading ────────────────────────────────────────────────────────
# Models are loaded on first request, not at startup.
# This prevents Railway from crashing before env vars are ready.

_retriever = None
_reranker  = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever(top_k=20)
    return _retriever


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


# ── Schemas ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class Citation(BaseModel):
    index:     int
    title:     str
    arxiv_url: str
    published: str


class QueryResponse(BaseModel):
    answer:    str
    citations: list[Citation]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    corpus_exists = Path("data/corpus.json").exists()
    return {
        "status":  "ok",
        "corpus":  corpus_exists,
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # 1. Hybrid retrieval
    candidates = get_retriever().retrieve(req.question)

    # 2. Reranking
    top_chunks = get_reranker().rerank(req.question, candidates, top_k=req.top_k)

    if not top_chunks:
        raise HTTPException(status_code=404, detail="No relevant chunks found.")

    # 3. Build citation-enforced prompt
    messages = build_prompt(req.question, top_chunks)

    # 4. Call LLM
    answer = call_llm(messages)

    # 5. Return answer + citations
    citations = format_citations(top_chunks)
    return QueryResponse(answer=answer, citations=citations)