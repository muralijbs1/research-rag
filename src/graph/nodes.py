from __future__ import annotations

from typing import Any, Optional, TypedDict

from src.config import RERANK_TOP_N, TOP_K_RETRIEVAL
from src.generation.generator import generate_answer
from src.generation.llm_router import generate
from src.generation.prompts_writer import (
    INTENT_CHECK_SYSTEM_PROMPT,
    INTENT_CHECK_USER_TEMPLATE,
    QUERY_REWRITE_TEMPLATE,
    GROQ_COMPARE_REJECTION_SYSTEM_PROMPT,
)
from src.retrieval.reranker import rerank
from src.retrieval.retriever import retrieve

MAX_RETRIES = 2


class RAGState(TypedDict, total=False):
    question: str
    original_question: str
    chunks: list[dict[str, Any]]
    reranked: list[dict[str, Any]]
    source_chunks: list[dict[str, Any]]
    answer: str
    retries: int
    quality_passed: bool
    error: Optional[str]


def intent_check_node(state: RAGState) -> RAGState:
    prompt = INTENT_CHECK_USER_TEMPLATE.format(question=state["question"])
    response = (generate(prompt, model="openai", temperature=0, system=INTENT_CHECK_SYSTEM_PROMPT) or "").strip()
    if "yes" in response.lower():
        return state
    
    rejection = (generate(
        f"User asked: {state['question']}",
        model="groq",
        system=GROQ_COMPARE_REJECTION_SYSTEM_PROMPT,
        temperature=0.9,
    ) or "This page is for research questions only!").strip()
    
    return {**state, "error": rejection}

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
    prompt = QUERY_REWRITE_TEMPLATE.format(question=state["question"])
    rewritten = (generate(prompt, model="openai") or "").strip()
    if not rewritten:
        return {**state, "error": "query rewrite returned empty result"}
    retries = int(state.get("retries") or 0)
    print(f"  [REWRITE] Original: {state['question']}")
    print(f"  [REWRITE] Rewritten: {rewritten}")
    return {**state, "question": rewritten, "retries": retries + 1}


def generate_node(state: RAGState) -> RAGState:
    reranked = state.get("reranked") or []
    if not reranked:
        return {**state, "error": "no reranked chunks to generate from"}
    answer = generate_answer(state["question"], reranked)
    return {**state, "answer": answer["answer"], "source_chunks": answer["source_chunks"]}


def should_retry(state: RAGState) -> str:
    if state.get("quality_passed"):
        return "generate"
    retries = int(state.get("retries") or 0)
    return "retry" if retries < MAX_RETRIES else "give_up"