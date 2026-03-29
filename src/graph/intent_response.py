"""
intent_response.py

Groq-powered conversational gateway for the chat page.

Responsibilities:
- Decide if a question is AI/ML research related
- If yes  → rewrite it as a standalone question and return route="rag"
- If no   → generate a friendly, varied response that steers the user toward research topics
- Generate a short conversation title from the first message
"""

from __future__ import annotations

import json
from typing import Any

from src.generation.llm_router import generate

AVAILABLE_PAPERS = [
    "Attention Is All You Need (Transformer architecture)",
    "Language Models are Few-Shot Learners (GPT-3)",
    "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    "ReAct: Synergizing Reasoning and Acting in Language Models",
    "Self-Consistency Improves Chain of Thought Reasoning",
    "Toolformer: Language Models Can Teach Themselves to Use Tools",
    "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation",
    "Mixture-of-Agents Enhances Large Language Model Capabilities",
    "TinyLlama: An Open-Source Small Language Model",
    "The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery",
    "Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG",
]

SYSTEM_PROMPT = f"""You are a conversational gateway for a Research Paper RAG system.

The system has these papers:
{chr(10).join(f"- {p}" for p in AVAILABLE_PAPERS)}

Your job is to analyze the conversation history and the current message, then respond with ONLY a JSON object.

If the message is related to AI, ML, deep learning, NLP, transformers, agents, RAG, LLMs, or the papers above:
- Rewrite the question as a complete standalone question that makes sense without any conversation history
- The rewritten question should be specific and self-contained — resolve any pronouns or references like "it", "that", "this", "them" using the conversation context
- Return: {{"route": "rag", "rewritten_question": "your rewritten standalone question here", "message": null}}
- If the user responds with short agreement or curiosity ("yeah", "tell me more", "explain", "go for it", "yes please", "sure", "ok", "and?", "interesting") AND the previous assistant message mentioned a specific paper or topic — treat it as related and rewrite it as a standalone question about that paper/topic. Route to RAG.


If the message is NOT related to research:
- Respond warmly and conversationally, steer toward the research papers
- Vary your tone and humor each time — never use the same response twice
- Mention specific papers when relevant
- Keep response under 3 sentences
- Return: {{"route": "chat", "rewritten_question": null, "message": "your friendly response here"}}

CRITICAL: Always return valid JSON only. No markdown, no extra text."""


def route_message(
    question: str,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Route and rewrite a user message through Groq.

    Parameters
    ----------
    question : str
        The user's current message.
    history : list[dict]
        Previous messages in format [{"role": "user"/"assistant", "content": "..."}]

    Returns
    -------
    dict with:
        - "route": "rag" or "chat"
        - "rewritten_question": str | None — standalone question for RAG
        - "message": str | None — response text if route="chat"
    """
    history_text = ""
    if history:
        last_6 = history[-6:]
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content'][:300]}" for m in last_6
        )
        history_text = f"Conversation history:\n{history_text}\n\n"

    prompt = f"{history_text}Current message: {question}"

    raw = generate(
        prompt,
        model="groq",
        system=SYSTEM_PROMPT,
        temperature=0.7,
    )

    try:
        clean = raw.strip().strip("```json").strip("```").strip()
        result = json.loads(clean)

        route = result.get("route")
        if route not in {"rag", "chat"}:
            return {"route": "rag", "rewritten_question": question, "message": None}

        return {
            "route": route,
            "rewritten_question": result.get("rewritten_question") or question,
            "message": result.get("message"),
        }
    except (json.JSONDecodeError, KeyError):
        return {"route": "rag", "rewritten_question": question, "message": None}


def generate_conversation_title(first_message: str) -> str:
    """
    Generate a short conversation title from the first user message.
    """
    prompt = f"Generate a short title (max 6 words) for a conversation that starts with: '{first_message}'. Return only the title, nothing else."

    try:
        title = generate(
            prompt,
            model="groq",
            system="You generate short descriptive conversation titles. Return only the title text, no quotes, no punctuation at the end.",
            temperature=0.5,
        )
        return title.strip()[:60]
    except Exception:
        return first_message[:40]