from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.graph.nodes import (
    MAX_RETRIES,
    RAGState,
    check_quality_node,
    generate_node,
    intent_check_node,
    rerank_node,
    retrieve_node,
    rewrite_query_node,
)


def after_intent(state: RAGState) -> str:
    if state.get("error"):
        return "end"
    return "retrieve"


def should_rewrite(state: RAGState) -> str:
    if state.get("error"):
        return "generate"
    retries = int(state.get("retries") or 0)
    if not state.get("quality_passed") and retries < MAX_RETRIES:
        return "rewrite"
    return "generate"


def build_graph() -> StateGraph:
    graph = StateGraph(RAGState)

    graph.add_node("intent_check", intent_check_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("check_quality", check_quality_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("intent_check")
    graph.add_conditional_edges("intent_check", after_intent, {
        "retrieve": "retrieve",
        "end": END,
    })
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "check_quality")
    graph.add_conditional_edges(
        "check_quality",
        should_rewrite,
        {"rewrite": "rewrite_query", "generate": "generate"},
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile()


rag_pipeline = build_graph()