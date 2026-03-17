"""
Central LLM router for the RAG system.

This is the only file in the project that should decide which LLM to use.
Every other file that needs an LLM response should call this module.
"""

from __future__ import annotations

from typing import Any, Optional

from litellm import completion

from src.config import ANTHROPIC_LITELLM_MODEL, DEFAULT_LITELLM, OPENAI_LITELLM_MODEL
from src.generation.prompt_builder import DEFAULT_SYSTEM_INSTRUCTION


def generate_with_metadata(
    prompt: str,
    *,
    model: Optional[str] = None,
    system: Optional[str] = None,
) -> dict[str, Any]:
    """
    Generate a response and return useful metadata.

    Returns
    -------
    dict
        - "text": final answer text
        - "model": resolved LiteLLM model name that was used
        - "token_count": total tokens if available, else None
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    resolved_model = resolve_model(model=model)
    if not resolved_model:
        raise ValueError("DEFAULT_LITELLM is empty; set it in your environment or src/config.py")

    response = completion(
        model=resolved_model,
        messages=[
            {"role": "system", "content": (system or DEFAULT_SYSTEM_INSTRUCTION)},
            {"role": "user", "content": prompt},
        ],
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
) -> str:
    """
    Generate a response using LiteLLM.

    If ``model`` is not provided, uses ``DEFAULT_LITELLM`` as a fail-safe.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    resolved_model = resolve_model(model=model)
    if not resolved_model:
        raise ValueError("DEFAULT_LITELLM is empty; set it in your environment or src/config.py")

    return generate_with_metadata(prompt, model=model, system=system)["text"]


def resolve_model(*, model: Optional[str] = None) -> str:
    """
    Return the model that will be used (no provider tracking).
    """
    model_input = (model or "").strip().lower()

    if model_input in {"openai"}:
        resolved_model = (OPENAI_LITELLM_MODEL or "").strip()
    elif model_input in {"anthropic", "claude"}:
        resolved_model = (ANTHROPIC_LITELLM_MODEL or "").strip()
    else:
        resolved_model = (model or "").strip() or (DEFAULT_LITELLM or "").strip()

    if not resolved_model:
        raise ValueError("DEFAULT_LITELLM is empty; set it in your environment or src/config.py")
    return resolved_model


# -----------------------------------------------------------------------------
# Direct SDK calls (OpenAI / Anthropic) using default models (commented out)
# -----------------------------------------------------------------------------


# import os
# from src.config import ANTHROPIC_MODEL, OPENAI_MODEL
# from src.generation.prompt_builder import DEFAULT_SYSTEM_INSTRUCTION
#
#
# def generate_openai(
#     prompt: str,
#     *,
#     model: Optional[str] = None,
#     system: Optional[str] = None,
# ) -> str:
#     if not prompt or not prompt.strip():
#         raise ValueError("prompt must be a non-empty string")
#
#     api_key = os.getenv("OPENAI_API_KEY")
#     if not api_key:
#         raise RuntimeError("OPENAI_API_KEY is not set")
#
#     try:
#         from openai import OpenAI  # type: ignore
#     except Exception as exc:
#         raise RuntimeError("OpenAI SDK not installed. `pip install openai`") from exc
#
#     client = OpenAI(api_key=api_key)
#     resolved_model = (model or "").strip() or OPENAI_MODEL
#     resp = client.chat.completions.create(
#         model=resolved_model,
#         messages=[
#             {"role": "system", "content": (system or DEFAULT_SYSTEM_INSTRUCTION)},
#             {"role": "user", "content": prompt},
#         ],
#     )
#     text = (resp.choices[0].message.content or "").strip()
#     if not text:
#         raise RuntimeError("OpenAI returned an empty response")
#     return text


# def generate_claude(
#     prompt: str,
#     *,
#     model: Optional[str] = None,
#     system: Optional[str] = None,
# ) -> str:
#     if not prompt or not prompt.strip():
#         raise ValueError("prompt must be a non-empty string")
#
#     api_key = os.getenv("ANTHROPIC_API_KEY")
#     if not api_key:
#         raise RuntimeError("ANTHROPIC_API_KEY is not set")
#
#     try:
#         from anthropic import Anthropic  # type: ignore
#     except Exception as exc:
#         raise RuntimeError("Anthropic SDK not installed. `pip install anthropic`") from exc
#
#     client = Anthropic(api_key=api_key)
#     resolved_model = (model or "").strip() or ANTHROPIC_MODEL
#     resp = client.messages.create(
#         model=resolved_model,
#         max_tokens=1024,
#         system=(system or DEFAULT_SYSTEM_INSTRUCTION),
#         messages=[{"role": "user", "content": prompt}],
#     )
#
#     parts: list[str] = []
#     for block in getattr(resp, "content", []) or []:
#         text = getattr(block, "text", None)
#         if isinstance(text, str) and text.strip():
#             parts.append(text.strip())
#     final = "\n\n".join(parts).strip()
#     if not final:
#         raise RuntimeError("Anthropic returned an empty response")
#     return final

