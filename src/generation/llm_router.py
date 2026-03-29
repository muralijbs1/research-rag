"""
Central LLM router for the RAG system.

This is the only file in the project that should decide which LLM to use.
Every other file that needs an LLM response should call this module.
"""

from __future__ import annotations

from typing import Any, Optional

from litellm import completion

from src.config import ANTHROPIC_LITELLM_MODEL, DEFAULT_LITELLM, GROQ_LITELLM_MODEL, OPENAI_LITELLM_MODEL
from src.generation.prompts_writer import RAG_SYSTEM_INSTRUCTION

def generate_with_metadata(
    prompt: str,
    *,
    model: Optional[str] = None,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
) -> dict[str, Any]:
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    resolved_model = resolve_model(model=model)
    if not resolved_model:
        raise ValueError("DEFAULT_LITELLM is empty; set it in your environment or src/config.py")

    response = completion(
        model=resolved_model,
        messages=[
            {"role": "system", "content": (system or RAG_SYSTEM_INSTRUCTION)},
            {"role": "user", "content": prompt},
        ],
        **({"temperature": temperature} if temperature is not None else {}),
    )

    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("LLM returned an empty response")

    usage = getattr(response, "usage", None)
    token_count: Optional[int] = None
    if isinstance(usage, dict):
        total = usage.get("total_tokens")
        if isinstance(total, int):
            token_count = total
    else:
        total = getattr(usage, "total_tokens", None)
        if isinstance(total, int):
            token_count = total

    return {"text": text, "model": resolved_model, "token_count": token_count}


def generate(
    prompt: str,
    *,
    model: Optional[str] = None,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
) -> str:
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    resolved_model = resolve_model(model=model)
    if not resolved_model:
        raise ValueError("DEFAULT_LITELLM is empty; set it in your environment or src/config.py")

    return generate_with_metadata(prompt, model=model, system=system, temperature=temperature)["text"]


def generate_stream(
    prompt: str,
    *,
    model: Optional[str] = None,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
):
    """
    Stream a response token by token.
    Yields string chunks as they arrive from the LLM.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    resolved_model = resolve_model(model=model)

    response = completion(
        model=resolved_model,
        messages=[
            {"role": "system", "content": (system or RAG_SYSTEM_INSTRUCTION)},
            {"role": "user", "content": prompt},
        ],
        stream=True,
        **({"temperature": temperature} if temperature is not None else {}),
    )

    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def resolve_model(*, model: Optional[str] = None) -> str:
    model_input = (model or "").strip().lower()

    if model_input in {"openai"}:
        resolved_model = (OPENAI_LITELLM_MODEL or "").strip()
    elif model_input in {"anthropic", "claude"}:
        resolved_model = (ANTHROPIC_LITELLM_MODEL or "").strip()
    elif model_input in {"groq"}:
        resolved_model = (GROQ_LITELLM_MODEL or "").strip()
    else:
        resolved_model = (model or "").strip() or (DEFAULT_LITELLM or "").strip()

    if not resolved_model:
        raise ValueError("DEFAULT_LITELLM is empty; set it in your environment or src/config.py")
    return resolved_model