"""Ingest, chunk, retrieve, and grow episodic memory."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from keepsake import db
from keepsake.db import MemoryRow
from keepsake.embeddings import embed_text
from keepsake.config import get_settings


def chunk_text(text: str, max_chars: int = 900) -> List[str]:
    text = text.strip()
    if not text:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    buf = ""
    for p in paragraphs:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = f"{buf}\n\n{p}".strip() if buf else p
        else:
            if buf:
                chunks.append(buf)
            if len(p) <= max_chars:
                buf = p
            else:
                for i in range(0, len(p), max_chars):
                    chunks.append(p[i : i + max_chars])
                buf = ""
    if buf:
        chunks.append(buf)
    return chunks


def ingest_text(
    content: str,
    *,
    kind: str = "story",
    title: str = "",
    people: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    created_at: Optional[datetime] = None,
) -> List[int]:
    ids: List[int] = []
    for i, chunk in enumerate(chunk_text(content)):
        chunk_title = title if i == 0 else f"{title} (part {i + 1})"
        emb = embed_text(f"{chunk_title}\n{chunk}")
        mid = db.insert_memory(
            kind=kind,
            title=chunk_title or kind,
            content=chunk,
            embedding=emb,
            people=people,
            tags=tags,
            created_at=created_at,
        )
        ids.append(mid)
    return ids


def ingest_file(path: Path, *, kind: str = "story") -> List[int]:
    text = path.read_text(encoding="utf-8")
    title = path.stem.replace("_", " ").title()
    people: List[str] = []
    tags: List[str] = []
    # Optional YAML-ish header: --- kind: recipe / people: A,B / tags: x,y ---
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            header = text[3:end]
            body = text[end + 3 :].strip()
            for line in header.splitlines():
                if ":" not in line:
                    continue
                k, v = line.split(":", 1)
                k, v = k.strip().lower(), v.strip()
                if k == "kind":
                    kind = v
                elif k == "title":
                    title = v
                elif k == "people":
                    people = [p.strip() for p in v.split(",") if p.strip()]
                elif k == "tags":
                    tags = [t.strip() for t in v.split(",") if t.strip()]
            text = body
    return ingest_text(text, kind=kind, title=title, people=people, tags=tags)


def retrieve(query: str, k: Optional[int] = None) -> List[MemoryRow]:
    settings = get_settings()
    k = k or settings.retrieve_k
    emb = embed_text(query)
    return db.hybrid_search(query, emb, k=k)


def store_exchange(user_text: str, assistant_text: str) -> Tuple[int, int]:
    """Persist the turn so the companion grows with every chat."""
    u = ingest_text(
        user_text,
        kind="event",
        title="Family question",
        tags=["conversation", "user"],
    )
    a = ingest_text(
        assistant_text,
        kind="event",
        title="Keepsake reply",
        tags=["conversation", "assistant"],
    )
    return (u[0] if u else -1, a[0] if a else -1)


def format_memories_for_prompt(rows: List[MemoryRow]) -> str:
    if not rows:
        return "(no memories retrieved yet)"
    blocks = []
    for r in rows:
        when = r.created_at.strftime("%Y-%m-%d") if r.created_at else "?"
        people = ", ".join(r.people) if r.people else "—"
        blocks.append(
            f"[{r.kind} | {when} | people: {people} | score={r.score:.4f}]\n"
            f"{r.title}\n{r.content}"
        )
    return "\n\n---\n\n".join(blocks)
