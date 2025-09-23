"""Logging helpers that default to JSON structured logs."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict

_DEFAULT_FORMATTER = None


class JsonFormatter(logging.Formatter):
    """Render log records as structured JSON."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401 - inherited docstring
        payload: Dict[str, Any] = {
            "ts": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key.startswith("_"):
                continue
            if key in payload:
                continue
            if key in {"args", "msg"}:
                continue
            payload[key] = value
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    global _DEFAULT_FORMATTER
    root = logging.getLogger()
    root.setLevel(level.upper())
    if not root.handlers:
        handler = logging.StreamHandler()
        _DEFAULT_FORMATTER = JsonFormatter()
        handler.setFormatter(_DEFAULT_FORMATTER)
        root.addHandler(handler)
    else:
        for handler in root.handlers:
            if _DEFAULT_FORMATTER is None:
                _DEFAULT_FORMATTER = JsonFormatter()
            handler.setFormatter(_DEFAULT_FORMATTER)


def get_logger(name: str) -> logging.Logger:
    if _DEFAULT_FORMATTER is None:
        configure_logging()
    return logging.getLogger(name)
