"""
Rerank retrieved chunks with either SBERT cross-encoder or Cohere Rerank API.

Controlled by the RERANKER config variable:
  RERANKER="sbert"  → cross-encoder/ms-marco-MiniLM-L-6-v2 (local, MPS-accelerated)
  RERANKER="cohere" → Cohere Rerank API (requires COHERE_API_KEY)
"""

import os
import time
from typing import Any

from sentence_transformers import CrossEncoder

from src.config import RERANKER

# SBERT cross-encoder — loaded once on first use
_sbert_model: CrossEncoder | None = None


def _get_sbert_model() -> CrossEncoder:
    global _sbert_model
    if _sbert_model is None:
        _sbert_model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device="mps",
        )
    return _sbert_model


def _rerank_sbert(question: str, chunks: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    model = _get_sbert_model()
    pairs = [[question, c.get("text", "") or ""] for c in chunks]
    scores = model.predict(pairs)
    scored = [{**chunk, "score": float(score)} for chunk, score in zip(chunks, scores)]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_n]


def _rerank_cohere(question: str, chunks: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    import cohere

    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("COHERE_API_KEY is not set in your environment")

    client = cohere.Client(api_key)
    docs = [c.get("text", "") or "" for c in chunks]

    time.sleep(6)  # Trial key: max 10 calls/min → wait 6s between calls
    response = client.rerank(
        model="rerank-english-v3.0",
        query=question,
        documents=docs,
        top_n=top_n,
    )

    results = []
    for hit in response.results:
        chunk = {**chunks[hit.index], "score": hit.relevance_score}
        results.append(chunk)
    return results


def rerank(
    question: str,
    chunks: list[dict[str, Any]],
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """
    Score each chunk against the question and return the top N by relevance.

    Routes to SBERT or Cohere based on the RERANKER config variable.
    Expects each chunk to have at least a "text" key.
    """
    if not question or not question.strip():
        raise ValueError("Question must be a non-empty string")
    if not chunks:
        return []
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    if RERANKER == "cohere":
        return _rerank_cohere(question, chunks, top_n)
    return _rerank_sbert(question, chunks, top_n)
