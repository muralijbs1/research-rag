from typing import Any

from src.ingestion.embedder import embed_texts
from src.retrieval.vector_store import VectorStore


def retrieve(
    question: str,
    top_k: int = 20,
    vector_store: VectorStore | None = None,
) -> list[dict[str, Any]]:
    """
    Embed ``question`` and return the top-k most similar chunks from ChromaDB.

    This is a stateless functional retriever:
    - embeds the input question
    - queries the persistent Chroma collection via ``VectorStore``
    - returns the top-k chunks with their scores
    """
    if not question or not question.strip():
        raise ValueError("Question must be a non‑empty string")

    store = vector_store or VectorStore()
    query_vectors = embed_texts([question])
    if not query_vectors:
        raise RuntimeError("Failed to compute embedding for question")

    return store.search(query_vector=query_vectors[0], top_k=top_k)