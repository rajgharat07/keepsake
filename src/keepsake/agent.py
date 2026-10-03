"""Agent loop: retrieve -> generate -> store, with Sentry spans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from keepsake import memory, llm
from keepsake.config import get_settings
from keepsake.db import MemoryRow
from keepsake import tracing


@dataclass
class ChatResult:
    reply: str
    memories: List[MemoryRow]
    tokens: dict
    stored: bool


def ask(question: str, *, store: bool = True) -> ChatResult:
    settings = get_settings()
    with tracing.agent_transaction("keepsake.chat"):
        rows = memory.retrieve(question)
        context = memory.format_memories_for_prompt(rows)
        system = llm.build_system_prompt(
            settings.keepsake_person_name,
            settings.keepsake_person_relation,
        )
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"Retrieved memories about {settings.keepsake_person_name}:\n\n"
                    f"{context}\n\n"
                    f"Question from the family:\n{question}"
                ),
            },
        ]
        reply, tokens = llm.chat(messages)
        stored = False
        if store:
            memory.store_exchange(question, reply)
            stored = True
        return ChatResult(reply=reply, memories=rows, tokens=tokens, stored=stored)


def on_this_day_blurb() -> Optional[str]:
    from keepsake import db

    rows = db.memories_on_this_day()
    if not rows:
        return None
    bits = []
    for r in rows:
        year = r.created_at.year if r.created_at else "?"
        bits.append(f"- ({year}) {r.title or r.kind}: {r.content[:180]}…")
    return "On this day in the family archive:\n" + "\n".join(bits)
