"""
Rerank retrieved chunks with a cross-encoder for true relevance to the question.

The retriever returns many chunks by embedding similarity; this module scores
each (question, chunk) pair with a cross-encoder and returns only the top N,
improving precision for the RAG pipeline.
"""

from typing import Any

from sentence_transformers import CrossEncoder

# Load once at import so we don't re-download or re-allocate on every rerank call.
# M3 Mac: use MPS for GPU acceleration.
_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    """Return the shared cross-encoder, loading it on first use."""
    global _model
    if _model is None:
        _model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device="mps",
        )
    return _model


def rerank(
    question: str,
    chunks: list[dict[str, Any]],
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """
    Score each chunk against the question and return the top N by relevance.

    Expects each chunk to have at least a "text" key (e.g. from the retriever).
    Builds [question, chunk_text] pairs, scores with the cross-encoder, sorts
    by score descending, and returns the top_n chunks (preserving dict shape;
    the original "score" is replaced with the reranker score).
    """
    if not question or not question.strip():
        raise ValueError("Question must be a non-empty string")
    if not chunks:
        return []
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    model = _get_model()
    pairs = [[question, c.get("text", "") or ""] for c in chunks]
    scores = model.predict(pairs)

    # scores is a 1D array; attach to chunks and sort descending (higher = more relevant)
    scored = [
        {**chunk, "score": float(score)}
        for chunk, score in zip(chunks, scores)
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_n]
