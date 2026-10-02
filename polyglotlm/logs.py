"""Structured logging. Every log line is an event name plus key/value fields."""

from __future__ import annotations

import logging
import os
import sys

import structlog


class _CurrentStderr:
    """Resolves sys.stderr on every write, so swapped streams (tests, CLI runners) keep working."""

    def write(self, text: str) -> int:
        return sys.stderr.write(text)

    def flush(self) -> None:
        sys.stderr.flush()


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog once. Set POLYGLOT_LOG_JSON=1 for machine-readable lines."""
    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer(ensure_ascii=False)
        if os.environ.get("POLYGLOT_LOG_JSON") == "1"
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        logger_factory=structlog.PrintLoggerFactory(file=_CurrentStderr()),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
