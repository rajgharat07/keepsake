"""Local embeddings via Ollama (nomic-embed-text)."""

from __future__ import annotations

from typing import List

import httpx

from keepsake.config import get_settings
from keepsake import tracing


def embed_text(text: str) -> List[float]:
    settings = get_settings()
    payload = {
        "model": settings.ollama_embed_model,
        "prompt": text.strip(),
    }
    with tracing.span("ai.embed", settings.ollama_embed_model):
        with httpx.Client(base_url=settings.ollama_host, timeout=120.0) as client:
            r = client.post("/api/embeddings", json=payload)
            r.raise_for_status()
            data = r.json()
    vec = data.get("embedding")
    if not vec:
        raise RuntimeError("Ollama returned empty embedding")
    return list(vec)


def embed_batch(texts: List[str]) -> List[List[float]]:
    return [embed_text(t) for t in texts if t.strip()]
