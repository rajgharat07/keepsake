"""Tiger Data layer: pgvector + full-text + Timescale hypertable."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterator, List, Optional, Sequence

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from keepsake.config import get_settings
from keepsake import tracing


@dataclass
class MemoryRow:
    id: int
    created_at: datetime
    kind: str
    title: str
    content: str
    people: List[str]
    tags: List[str]
    score: float = 0.0


def _normalize_dsn(url: str) -> str:
    if url.startswith("postgres://"):
        return "postgresql://" + url[len("postgres://") :]
    return url


@contextmanager
def connect(*, register: bool = True) -> Iterator[psycopg.Connection]:
    settings = get_settings()
    if not settings.database_url.strip():
        raise RuntimeError(
            "DATABASE_URL is empty. Copy .env.example to .env and set your Tiger connection string."
        )
    conn = psycopg.connect(
        _normalize_dsn(settings.database_url),
        row_factory=dict_row,
        autocommit=False,
    )
    try:
        if register:
            register_vector(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_schema() -> None:
    # Extension must exist before pgvector's register_vector runs
    with connect(register=False) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
            except Exception:
                pass

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id              BIGSERIAL,
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    kind            TEXT NOT NULL DEFAULT 'story',
                    title           TEXT NOT NULL DEFAULT '',
                    content         TEXT NOT NULL,
                    people          TEXT[] NOT NULL DEFAULT '{}',
                    tags            TEXT[] NOT NULL DEFAULT '{}',
                    embedding       vector(768),
                    content_tsv     tsvector,
                    PRIMARY KEY (id, created_at)
                );
                """
            )

            try:
                cur.execute(
                    """
                    SELECT create_hypertable(
                        'memories',
                        by_range('created_at'),
                        if_not_exists => TRUE,
                        migrate_data => TRUE
                    );
                    """
                )
            except Exception:
                try:
                    cur.execute(
                        """
                        SELECT create_hypertable(
                            'memories',
                            'created_at',
                            if_not_exists => TRUE,
                            migrate_data => TRUE
                        );
                        """
                    )
                except Exception:
                    pass

            try:
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS memories_embedding_hnsw
                        ON memories USING hnsw (embedding vector_cosine_ops);
                    """
                )
            except Exception:
                try:
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS memories_embedding_ivfflat
                            ON memories USING ivfflat (embedding vector_cosine_ops)
                            WITH (lists = 100);
                        """
                    )
                except Exception:
                    pass

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS memories_content_tsv_gin
                    ON memories USING gin (content_tsv);
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS memories_kind_idx ON memories (kind);"
            )


def insert_memory(
    *,
    kind: str,
    title: str,
    content: str,
    embedding: Sequence[float],
    people: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    created_at: Optional[datetime] = None,
) -> int:
    people = people or []
    tags = tags or []
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO memories (
                    created_at, kind, title, content, people, tags, embedding, content_tsv
                ) VALUES (
                    COALESCE(%s, NOW()),
                    %s, %s, %s, %s, %s, %s,
                    to_tsvector('english', coalesce(%s, '') || ' ' || coalesce(%s, ''))
                )
                RETURNING id;
                """,
                (
                    created_at,
                    kind,
                    title,
                    content,
                    people,
                    tags,
                    list(embedding),
                    title,
                    content,
                ),
            )
            row = cur.fetchone()
            return int(row["id"])


def count_memories() -> int:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS c FROM memories;")
            return int(cur.fetchone()["c"])


def hybrid_search(
    query: str,
    query_embedding: Sequence[float],
    *,
    k: int = 6,
) -> List[MemoryRow]:
    """Reciprocal rank fusion of vector cosine + full-text rank."""
    with tracing.span("db.hybrid_search", f"k={k}"):
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    WITH vec AS (
                        SELECT
                            id,
                            created_at,
                            kind,
                            title,
                            content,
                            people,
                            tags,
                            1 - (embedding <=> %s::vector) AS vscore,
                            ROW_NUMBER() OVER (
                                ORDER BY embedding <=> %s::vector
                            ) AS vrank
                        FROM memories
                        WHERE embedding IS NOT NULL
                        ORDER BY embedding <=> %s::vector
                        LIMIT 40
                    ),
                    fts AS (
                        SELECT
                            id,
                            created_at,
                            kind,
                            title,
                            content,
                            people,
                            tags,
                            ts_rank_cd(
                                content_tsv,
                                plainto_tsquery('english', %s)
                            ) AS tscore,
                            ROW_NUMBER() OVER (
                                ORDER BY ts_rank_cd(
                                    content_tsv,
                                    plainto_tsquery('english', %s)
                                ) DESC
                            ) AS trank
                        FROM memories
                        WHERE content_tsv @@ plainto_tsquery('english', %s)
                           OR content ILIKE '%%' || %s || '%%'
                        ORDER BY tscore DESC NULLS LAST
                        LIMIT 40
                    ),
                    fused AS (
                        SELECT
                            COALESCE(v.id, f.id) AS id,
                            COALESCE(v.created_at, f.created_at) AS created_at,
                            COALESCE(v.kind, f.kind) AS kind,
                            COALESCE(v.title, f.title) AS title,
                            COALESCE(v.content, f.content) AS content,
                            COALESCE(v.people, f.people) AS people,
                            COALESCE(v.tags, f.tags) AS tags,
                            COALESCE(1.0 / (60 + v.vrank), 0)
                              + COALESCE(1.0 / (60 + f.trank), 0) AS rrf
                        FROM vec v
                        FULL OUTER JOIN fts f
                          ON v.id = f.id AND v.created_at = f.created_at
                    )
                    SELECT * FROM fused
                    ORDER BY rrf DESC
                    LIMIT %s;
                    """,
                    (
                        list(query_embedding),
                        list(query_embedding),
                        list(query_embedding),
                        query,
                        query,
                        query,
                        query[:80],
                        k,
                    ),
                )
                rows = cur.fetchall()
    return [
        MemoryRow(
            id=int(r["id"]),
            created_at=r["created_at"],
            kind=r["kind"],
            title=r["title"] or "",
            content=r["content"],
            people=list(r["people"] or []),
            tags=list(r["tags"] or []),
            score=float(r["rrf"] or 0),
        )
        for r in rows
    ]


def memories_on_this_day(ref: Optional[date] = None, limit: int = 5) -> List[MemoryRow]:
    ref = ref or date.today()
    with connect() as conn:
        with conn.cursor() as cur:
            # Prefer archived / non-conversation rows so demos show past years
            cur.execute(
                """
                SELECT
                    id, created_at, kind, title, content, people, tags,
                    1.0 AS score
                FROM memories
                WHERE EXTRACT(MONTH FROM created_at) = %s
                  AND EXTRACT(DAY FROM created_at) = %s
                  AND NOT ('conversation' = ANY (tags))
                ORDER BY created_at ASC
                LIMIT %s;
                """,
                (ref.month, ref.day, limit),
            )
            rows = cur.fetchall()
            if not rows:
                cur.execute(
                    """
                    SELECT
                        id, created_at, kind, title, content, people, tags,
                        1.0 AS score
                    FROM memories
                    WHERE EXTRACT(MONTH FROM created_at) = %s
                      AND EXTRACT(DAY FROM created_at) = %s
                    ORDER BY created_at ASC
                    LIMIT %s;
                    """,
                    (ref.month, ref.day, limit),
                )
                rows = cur.fetchall()
    return [
        MemoryRow(
            id=int(r["id"]),
            created_at=r["created_at"],
            kind=r["kind"],
            title=r["title"] or "",
            content=r["content"],
            people=list(r["people"] or []),
            tags=list(r["tags"] or []),
            score=1.0,
        )
        for r in rows
    ]


def healthcheck() -> dict:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version() AS v;")
            version = cur.fetchone()["v"]
            cur.execute(
                "SELECT COUNT(*) AS c FROM pg_extension WHERE extname IN ('vector', 'timescaledb');"
            )
            ext_count = int(cur.fetchone()["c"])
            cur.execute("SELECT COUNT(*) AS c FROM memories;")
            n = int(cur.fetchone()["c"])
    return {"postgres": version.split(",")[0], "extensions_seen": ext_count, "memories": n}
