"""Sentry agent tracing — spans around embed / retrieve / generate."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

import sentry_sdk
from sentry_sdk import start_span, start_transaction

from keepsake.config import get_settings

_initialized = False


def init_tracing() -> bool:
    global _initialized
    if _initialized:
        return True
    dsn = get_settings().sentry_dsn.strip()
    if not dsn:
        return False
    sentry_sdk.init(
        dsn=dsn,
        traces_sample_rate=1.0,
        profiles_sample_rate=0.0,
        send_default_pii=False,
        environment="keepsake-local",
    )
    _initialized = True
    return True


@contextmanager
def agent_transaction(name: str = "keepsake.chat") -> Iterator[Optional[object]]:
    if not init_tracing():
        yield None
        return
    with start_transaction(op="agent", name=name) as txn:
        yield txn


@contextmanager
def span(op: str, description: str) -> Iterator[Optional[object]]:
    if not _initialized:
        yield None
        return
    with start_span(op=op, description=description) as s:
        yield s
