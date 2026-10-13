"""Structured logging (structlog). Transcripts, notes and identifiers must never reach a log line: the code logs ids,
counts, timings and model ids, and a last-line-of-defence processor drops keys that commonly carry text."""

from __future__ import annotations

import logging
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog

_SENSITIVE_KEYS = frozenset(
    {"text", "content", "messages", "body", "prompt", "answer", "transcript", "note", "quote", "value", "name", "dob"}
)


def _drop_sensitive(_: Any, __: str, event_dict: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    for key in _SENSITIVE_KEYS & event_dict.keys():
        event_dict[key] = "[dropped]"
    return event_dict


def _stderr_logger(*_: Any) -> structlog.PrintLogger:
    return structlog.PrintLogger(file=sys.stderr)


def configure_logging(level: str = "INFO", fmt: str = "console") -> None:
    shared: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        _drop_sensitive,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    renderers: list[structlog.types.Processor] = (
        [structlog.processors.format_exc_info, structlog.processors.JSONRenderer()]
        if fmt == "json"
        else [structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())]
    )
    structlog.configure(
        processors=[*shared, *renderers],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level.upper())),
        logger_factory=_stderr_logger,
        cache_logger_on_first_use=False,
    )
    logging.basicConfig(level=level.upper(), stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore", "urllib3", "faster_whisper"):
        logging.getLogger(noisy).setLevel(max(logging.getLevelName(level.upper()), logging.WARNING))


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]
