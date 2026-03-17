from __future__ import annotations

from typing import Any, Optional

from src.generation.llm_router import generate_with_metadata
from src.generation.prompt_builder import build_prompt


def generate_answer(
    question: str,
    reranked_chunks: list[dict[str, Any]],
    *,
    top_n: int = 5,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """
    End-to-end "generation" step for the RAG pipeline.

    Why this exists
    ---------------
    Downstream callers (Streamlit UI, evaluation, notebooks) should have a single,
    clean entrypoint that:
    - builds a grounded prompt from the question + reranked context chunks
    - calls the project's central LLM router
    - returns a consistent payload for UI + eval (answer + provenance + metadata)

    Parameters
    ----------
    question:
        The user's question.
    reranked_chunks:
        Chunks sorted by relevance (highest first). Each chunk should contain at
        least ``{"text": str}``.
    top_n:
        How many chunks to include in the prompt and report as "used".
    model:
        Optional model selector passed through to `llm_router` (e.g. "openai",
        "anthropic", or a full LiteLLM model string).

    Returns
    -------
    dict
        - "answer": str
        - "source_chunks": list[dict[str, Any]] (the chunks actually included)
        - "model": str (resolved LiteLLM model string)
        - "token_count": int | None
    """
    used_chunks = (reranked_chunks or [])[:top_n]
    prompt = build_prompt(question=question, chunks=used_chunks, top_n=len(used_chunks) if used_chunks else top_n)

    result = generate_with_metadata(prompt, model=model)
    return {
        "answer": result["text"],
        "source_chunks": used_chunks,
        "model": result["model"],
        "token_count": result["token_count"],
    }
