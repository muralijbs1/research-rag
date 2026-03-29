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

from src.generation.prompts_writer import GROQ_GATEWAY_SYSTEM_PROMPT, GROQ_TITLE_SYSTEM_PROMPT, GROQ_TITLE_USER_TEMPLATE



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
        system=GROQ_GATEWAY_SYSTEM_PROMPT,
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
    prompt = GROQ_TITLE_USER_TEMPLATE.format(message=first_message)

    try:
        title = generate(
            prompt,
            model="groq",
            system=GROQ_TITLE_SYSTEM_PROMPT,
            temperature=0.5,
        )
        return title.strip()[:60]
    except Exception:
        return first_message[:40]