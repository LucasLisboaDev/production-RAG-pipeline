"""
FastAPI Application (Week 3)
==============================
Ties together the full RAG pipeline into a REST API.

Run:
    uvicorn api.main:app --reload

Endpoints:
    POST /query   — Ask a question, get a cited answer
    GET  /health  — Health check
"""

from fastapi import FastAPI, HTTPException
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

# Load models once at startup
retriever = HybridRetriever(top_k=20)
reranker  = Reranker()


# ── Schemas ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5   # final chunks passed to LLM after reranking


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
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # 1. Hybrid retrieval (BM25 + vector + RRF)
    candidates = retriever.retrieve(req.question)

    # 2. Cross-encoder reranking
    top_chunks = reranker.rerank(req.question, candidates, top_k=req.top_k)

    if not top_chunks:
        raise HTTPException(status_code=404, detail="No relevant chunks found.")

    # 3. Build citation-enforced prompt
    messages = build_prompt(req.question, top_chunks)

    # 4. Call LLM
    answer = call_llm(messages)

    # 5. Return answer + citations
    citations = format_citations(top_chunks)
    return QueryResponse(answer=answer, citations=citations)
