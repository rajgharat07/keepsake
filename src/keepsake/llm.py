"""Local chat via Ollama (Gemma 3)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import httpx

from keepsake.config import get_settings
from keepsake import tracing


def chat(
    messages: List[Dict[str, str]],
    *,
    temperature: float = 0.4,
) -> Tuple[str, Dict[str, int]]:
    """Return (reply_text, token_stats)."""
    settings = get_settings()
    payload = {
        "model": settings.ollama_chat_model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    with tracing.span("ai.generate", settings.ollama_chat_model) as s:
        with httpx.Client(base_url=settings.ollama_host, timeout=300.0) as client:
            r = client.post("/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
        prompt_eval = int(data.get("prompt_eval_count") or 0)
        eval_count = int(data.get("eval_count") or 0)
        if s is not None:
            s.set_data("prompt_tokens", prompt_eval)
            s.set_data("completion_tokens", eval_count)
            s.set_data("model", settings.ollama_chat_model)

    msg = data.get("message") or {}
    content = (msg.get("content") or "").strip()
    if not content:
        raise RuntimeError("Ollama returned empty chat response")
    return content, {
        "prompt_tokens": prompt_eval,
        "completion_tokens": eval_count,
        "total_tokens": prompt_eval + eval_count,
    }


def build_system_prompt(person_name: str, relation: str) -> str:
    return (
        f"You are Keepsake, a warm private memory companion for {person_name} "
        f"({relation}). You only know what is in the retrieved memories below. "
        "Speak gently and concretely. Prefer quoting or paraphrasing real memories "
        "over inventing details. If something is not in memory, say you do not "
        "remember yet and invite the family to add it. Never claim medical advice. "
        "Keep replies short unless the user asks for a story."
    )
