"""
Qdrant Client Factory
======================
Shared connection to Qdrant Cloud.
Reads QDRANT_URL and QDRANT_API_KEY from .env

Usage:
    from retrieval.qdrant_client_factory import get_client, COLLECTION_NAME
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

COLLECTION_NAME = "arxiv_rag"
VECTOR_SIZE     = 384   # matches BAAI/bge-small-en-v1.5


def get_client() -> QdrantClient:
    url     = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url or not api_key:
        raise ValueError(
            "QDRANT_URL and QDRANT_API_KEY must be set in your .env file"
        )

    return QdrantClient(url=url, api_key=api_key)
