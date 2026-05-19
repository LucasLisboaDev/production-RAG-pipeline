"""
ArXiv Ingestion Pipeline for RAG
=================================
Downloads papers from ArXiv, extracts text via PyMuPDF,
chunks with overlap, and saves a JSON corpus ready for embedding.

Usage:
    python ingestion/ingest_arxiv.py

Output:
    data/corpus.json   — list of chunk dicts
    data/papers/       — raw PDFs (kept for reference)

Requirements:
    pip install arxiv pymupdf tqdm
"""

import arxiv
import certifi
import fitz  # PyMuPDF
import json
import os
import re
import ssl
import time
import urllib.request
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

SEARCH_QUERY    = "retrieval augmented generation RAG"
MAX_PAPERS      = 50
CHUNK_SIZE      = 512   # words per chunk
CHUNK_OVERLAP   = 64    # word overlap between chunks
MIN_CHUNK_WORDS = 30

DATA_DIR   = Path("data")
PDF_DIR    = DATA_DIR / "papers"
CORPUS_OUT = DATA_DIR / "corpus.json"

PDF_DIR.mkdir(parents=True, exist_ok=True)


# ── Step 1: Search & Download ─────────────────────────────────────────────────

def fetch_papers(query: str, max_results: int) -> list[arxiv.Result]:
    print(f"\n[1/3] Searching ArXiv: '{query}' (max {max_results} papers)")
    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    results = list(client.results(search))
    print(f"      Found {len(results)} papers")
    return results


def _download_pdf(url: str, dest: Path) -> None:
    """Download a PDF with certifi's CA bundle (fixes macOS python.org SSL errors)."""
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=ctx) as resp:
        dest.write_bytes(resp.read())


def download_pdfs(papers: list[arxiv.Result]) -> list[dict]:
    metadata = []
    print(f"\n[2/3] Downloading PDFs to {PDF_DIR}/")

    for paper in tqdm(papers, desc="Downloading"):
        paper_id = paper.entry_id.split("/")[-1]
        pdf_path = PDF_DIR / f"{paper_id}.pdf"

        if not pdf_path.exists():
            try:
                _download_pdf(paper.pdf_url, pdf_path)
                time.sleep(1)
            except Exception as e:
                print(f"      SKIP {paper_id}: {e}")
                continue

        metadata.append({
            "paper_id":  paper_id,
            "title":     paper.title,
            "authors":   [a.name for a in paper.authors],
            "published": paper.published.strftime("%Y-%m-%d"),
            "abstract":  paper.summary.replace("\n", " "),
            "arxiv_url": paper.entry_id,
            "pdf_path":  str(pdf_path),
        })

    print(f"      Downloaded {len(metadata)} PDFs")
    return metadata


# ── Step 2: PDF → Clean Text ──────────────────────────────────────────────────

def extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages_text = []

    for page in doc:
        text = page.get_text("text")
        if len(text.strip()) < 100:
            continue
        if re.search(r"^\s*references\s*$", text, re.IGNORECASE | re.MULTILINE):
            break
        pages_text.append(text)

    doc.close()
    full_text = "\n".join(pages_text)
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)
    full_text = re.sub(r"(?<!\n)\n(?!\n)", " ", full_text)
    full_text = re.sub(r"\s{2,}", " ", full_text)
    return full_text.strip()


# ── Step 3: Chunking ──────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if len(chunk.split()) >= MIN_CHUNK_WORDS:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def build_corpus():
    papers   = fetch_papers(SEARCH_QUERY, MAX_PAPERS)
    metadata = download_pdfs(papers)

    print(f"\n[3/3] Extracting text and chunking")
    corpus  = []
    skipped = 0

    for meta in tqdm(metadata, desc="Processing"):
        try:
            text = extract_text(meta["pdf_path"])
        except Exception as e:
            print(f"      SKIP {meta['paper_id']}: {e}")
            skipped += 1
            continue

        if len(text.split()) < 200:
            skipped += 1
            continue

        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

        for i, chunk in enumerate(chunks):
            corpus.append({
                "chunk_id":     f"{meta['paper_id']}_chunk_{i:04d}",
                "text":         chunk,
                "title":        meta["title"],
                "authors":      meta["authors"],
                "published":    meta["published"],
                "arxiv_url":    meta["arxiv_url"],
                "abstract":     meta["abstract"],
                "paper_id":     meta["paper_id"],
                "chunk_index":  i,
                "total_chunks": len(chunks),
            })

    with open(CORPUS_OUT, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)

    print(f"\n Done!")
    print(f"   Papers processed : {len(metadata) - skipped}")
    print(f"   Papers skipped   : {skipped}")
    print(f"   Total chunks     : {len(corpus)}")
    print(f"   Corpus saved to  : {CORPUS_OUT}")
    print(f"\n   Next step: run ingestion/embed_corpus.py")
    return corpus


if __name__ == "__main__":
    build_corpus()
