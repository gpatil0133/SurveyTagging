"""Shared logging configuration.

The whole codebase logs structured events as `logger.debug("event_name",
extra={...})`. The stdlib default formatter only renders `%(message)s`, so all
those `extra` key/values are silently dropped — you see `stage_start` but not
*which* stage. `ExtraFormatter` appends the non-standard LogRecord attributes
(i.e. everything passed via `extra=`) to the end of each line as `key=value`.
"""

from __future__ import annotations

import logging
import sys

# All attributes the stdlib puts on a LogRecord by default. Anything *else* on
# the record came from a caller's `extra={...}` and should be rendered.
_RESERVED = set(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
) | {"message", "asctime", "taskName"}

_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


class ExtraFormatter(logging.Formatter):
    """Formatter that appends `extra={...}` fields as ` | k=v k=v`."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _RESERVED and not k.startswith("_")
        }
        if not extras:
            return base
        kv = " ".join(f"{k}={v!r}" for k, v in extras.items())
        return f"{base} | {kv}"


def configure_logging(level: str = "INFO", stream=sys.stderr) -> None:
    """Install the ExtraFormatter on the root logger.

    Idempotent-ish: replaces the formatter on the root logger's handlers, or
    adds a handler if none exist. Safe to call once at process start.
    """
    log_level = getattr(logging, str(level).upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(log_level)

    formatter = ExtraFormatter(_FORMAT)
    if root.handlers:
        for handler in root.handlers:
            handler.setFormatter(formatter)
            handler.setLevel(log_level)
    else:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        root.addHandler(handler)
