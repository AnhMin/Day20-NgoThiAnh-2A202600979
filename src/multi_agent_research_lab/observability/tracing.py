"""Tracing hooks.

Supports a minimal local span recorder and optional LangSmith integration.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)

_SPANS: list[dict[str, Any]] = []


def get_recorded_spans() -> list[dict[str, Any]]:
    """Return spans recorded during the current process (for tests/debug)."""

    return list(_SPANS)


def clear_recorded_spans() -> None:
    _SPANS.clear()


def _langsmith_enabled() -> bool:
    return bool(os.getenv("LANGSMITH_API_KEY"))


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Record a span locally and optionally forward to LangSmith."""

    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}

    if _langsmith_enabled():
        try:
            from langsmith import traceable

            @traceable(name=name, metadata=attributes or {})
            def _noop() -> None:
                return None

            _noop()
        except Exception as exc:  # noqa: BLE001
            logger.debug("LangSmith tracing unavailable: %s", exc)

    try:
        yield span
    finally:
        span["duration_seconds"] = perf_counter() - started
        _SPANS.append(span)
        logger.debug("span=%s duration=%.3fs attrs=%s", name, span["duration_seconds"], attributes)
