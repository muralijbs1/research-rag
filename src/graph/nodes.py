from __future__ import annotations

from typing import Any, Optional, TypedDict

from src.config import RERANK_TOP_N, TOP_K_RETRIEVAL
from src.generation.generator import generate_answer
from src.generation.llm_router import generate
from src.retrieval.reranker import rerank
from src.retrieval.retriever import retrieve

MAX_RETRIES = 2


class RAGState(TypedDict, total=False):
    question: str
    original_question: str
    chunks: list[dict[str, Any]]
    reranked: list[dict[str, Any]]
    answer: dict[str, Any]
    retries: int
    quality_passed: bool
    error: Optional[str]


def intent_check_node(state: RAGState) -> RAGState:
    prompt = (
        "Is this question related to artificial intelligence, machine learning, or academic research topics? "
        "Reply with only 'yes' or 'no'.\n\n"
        f"Question: {state['question']}"
    )
    response = (generate(prompt, model="openai") or "").strip()
    if "yes" in response.lower():
        return state
    return {**state, "error": "I can only answer questions about AI/ML research papers. Please ask me something related to the papers in our library!"}


def retrieve_node(state: RAGState) -> RAGState:
    if not state.get("question", "").strip():
        return {**state, "error": "question is empty"}
    chunks = retrieve(state["question"], top_k=TOP_K_RETRIEVAL)
    if not chunks:
        return {**state, "error": "no chunks found for question"}
    return {**state, "chunks": chunks}


def rerank_node(state: RAGState) -> RAGState:
    chunks = state.get("chunks") or []
    if not chunks:
        return {**state, "error": "no chunks to rerank"}
    reranked = rerank(state["question"], chunks, top_n=min(RERANK_TOP_N, len(chunks)))
    return {**state, "reranked": reranked}


def check_quality_node(state: RAGState) -> RAGState:
    reranked = state.get("reranked") or []
    if not reranked:
        return {**state, "quality_passed": False}
    avg_score = sum(c.get("score", 0.0) for c in reranked) / len(reranked)
    return {**state, "quality_passed": avg_score > 0}


def rewrite_query_node(state: RAGState) -> RAGState:
    prompt = (
        "Rephrase this question using more specific technical terms from AI/ML research. "
        "Stay very close to the original meaning. Return only the rephrased question, nothing else.\n\n"
        f"Original: {state['question']}\nRephrased:"
    )
    rewritten = (generate(prompt, model="openai") or "").strip()
    if not rewritten:
        return {**state, "error": "query rewrite returned empty result"}
    retries = int(state.get("retries") or 0)
    return {**state, "question": rewritten, "retries": retries + 1}


def generate_node(state: RAGState) -> RAGState:
    reranked = state.get("reranked") or []
    if not reranked:
        return {**state, "error": "no reranked chunks to generate from"}
    answer = generate_answer(state["question"], reranked)
    return {**state, "answer": answer}


def should_retry(state: RAGState) -> str:
    if state.get("quality_passed"):
        return "generate"
    retries = int(state.get("retries") or 0)
    return "retry" if retries < MAX_RETRIES else "give_up"