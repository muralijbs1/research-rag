from __future__ import annotations

from typing import Any


DEFAULT_SYSTEM_INSTRUCTION: str = (
    "You are a helpful research assistant. Stay grounded in the provided context only. "
    "If the context does not contain enough information to answer, say you don't know "
    "and briefly describe what is missing. Do not fabricate citations or details."
)


def build_prompt(question: str, chunks: list[dict[str, Any]], *, top_n: int = 5) -> str:
    """
    Assemble the full prompt string sent to the LLM.

    What it includes (in order):
    - A system-style instruction to stay grounded in the provided context
    - The top ``top_n`` reranked chunks, cleanly formatted
    - The user's question

    Parameters
    ----------
    question:
        The user's question.
    chunks:
        Reranked chunks. Each chunk should contain at least ``{"text": str}``.
    top_n:
        Number of chunks to include from the start of the list.

    Returns
    -------
    str
        A single prompt string with consistent sections and formatting.
    """
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    selected_chunks = (chunks or [])[:top_n]

    formatted_chunks: list[str] = []
    for i, chunk in enumerate(selected_chunks, start=1):
        text = (chunk.get("text") or "").strip()
        formatted_chunks.append(
            "\n".join(
                [
                    f"[Chunk {i}]",
                    text if text else "(empty chunk text)",
                ]
            )
        )

    context_block = "\n\n".join(formatted_chunks) if formatted_chunks else "(no retrieved context)"

    return "\n\n".join(
        [
            "SYSTEM INSTRUCTION",
            DEFAULT_SYSTEM_INSTRUCTION,
            "CONTEXT (top reranked chunks)",
            context_block,
            "USER QUESTION",
            question.strip(),
        ]
    ).strip()
