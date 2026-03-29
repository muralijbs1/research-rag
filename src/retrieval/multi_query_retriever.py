"""
multi_query_retriever.py

Generates multiple query variants using Groq and retrieves chunks for each,
then merges and deduplicates results for richer retrieval coverage.
"""

from __future__ import annotations

import json
from typing import Any

from src.generation.llm_router import generate
from src.ingestion.embedder import embed_texts
from src.retrieval.vector_store import VectorStore
from src.generation.prompts_writer import GROQ_VARIANT_SYSTEM_PROMPT




def generate_query_variants(question: str) -> list[str]:
    """
    Generate 2 alternative phrasings of the question using Groq.
    Returns list of variants (not including the original).
    """
    try:
        raw = generate(
            f"Question: {question}",
            model="groq",
            system=GROQ_VARIANT_SYSTEM_PROMPT,
            temperature=0.7,
        )
        clean = raw.strip().strip("```json").strip("```").strip()
        variants = json.loads(clean)
        if isinstance(variants, list):
            return [v for v in variants if isinstance(v, str) and v.strip()][:2]
    except Exception:
        pass
    return []


def multi_query_retrieve(
    question: str,
    top_k: int = 20,
    vector_store: VectorStore | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve chunks using multiple query variants for better coverage.

    Parameters
    ----------
    question : str
        The main question (already rewritten by Groq if needed).
    top_k : int
        Number of chunks to retrieve per query variant.
    vector_store : VectorStore | None
        Optional VectorStore instance. Creates one if not provided.

    Returns
    -------
    list[dict] — deduplicated chunks from all query variants,
    ready for reranking. Each chunk has 'text', 'score', 'paper_name'.
    """
    store = vector_store or VectorStore()

    # Generate variants
    variants = generate_query_variants(question)
    all_queries = [question] + variants

    # Embed all queries
    embeddings = embed_texts(all_queries)

    # Retrieve for each query and merge
    seen_texts: set[str] = set()
    merged_chunks: list[dict[str, Any]] = []

    for embedding in embeddings:
        chunks = store.search(query_vector=embedding, top_k=top_k)
        for chunk in chunks:
            text = chunk.get("text", "")
            if text and text not in seen_texts:
                seen_texts.add(text)
                merged_chunks.append(chunk)

    return merged_chunks