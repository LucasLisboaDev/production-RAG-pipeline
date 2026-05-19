"""
Prompt Builder with Citation Enforcement (Week 3)
===================================================
Constructs the system + user prompt that forces the LLM
to only use provided context and cite sources by index.
"""


def build_prompt(query: str, chunks: list[dict]) -> list[dict]:
    """
    Returns an OpenAI-style messages list.
    Each chunk is numbered [1], [2], ... in the context block.
    The system prompt enforces citation and prohibits hallucination.
    """

    # Build numbered context block
    context_lines = []
    for i, chunk in enumerate(chunks, start=1):
        context_lines.append(
            f"[{i}] Title: {chunk['title']} ({chunk['published']})\n"
            f"    URL: {chunk['arxiv_url']}\n"
            f"    Excerpt: {chunk['text']}\n"
        )
    context_block = "\n".join(context_lines)

    system_prompt = (
        "You are a research assistant that answers questions strictly based on "
        "the provided research paper excerpts.\n\n"
        "Rules:\n"
        "1. Only use information from the excerpts below. Do NOT use outside knowledge.\n"
        "2. Every claim must be followed by a citation in the format [N] where N is "
        "the excerpt number.\n"
        "3. If the answer cannot be found in the excerpts, say: "
        "'I could not find this in the provided papers.'\n"
        "4. At the end of your answer, include a 'Sources' section listing every "
        "cited excerpt with its title and URL.\n"
    )

    user_prompt = (
        f"## Research Paper Excerpts\n\n{context_block}\n\n"
        f"## Question\n{query}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]


def format_citations(chunks: list[dict]) -> list[dict]:
    """Returns a deduplicated list of citation objects for the API response."""
    seen = set()
    citations = []
    for i, chunk in enumerate(chunks, start=1):
        if chunk["paper_id"] not in seen:
            seen.add(chunk["paper_id"])
            citations.append({
                "index":     i,
                "title":     chunk["title"],
                "arxiv_url": chunk["arxiv_url"],
                "published": chunk["published"],
            })
    return citations
