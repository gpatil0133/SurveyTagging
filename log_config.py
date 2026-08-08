"""Shared logging configuration.

Two sinks, deliberately different in kind:

  logs/app.log    rotating TEXT log — everything this process and uvicorn say.
                  For reading when something is wrong.
  logs/usage-*.jsonl  structured ledger — see usage_log.py.
                  For summing cost and latency. Never parse app.log for those.

The whole codebase logs structured events as `logger.debug("event_name",
extra={...})`. The stdlib default formatter only renders `%(message)s`, so all
those `extra` key/values are silently dropped — you see `stage_start` but not
*which* stage. `ExtraFormatter` appends the non-standard LogRecord attributes
(i.e. everything passed via `extra=`) to the end of each line as `key=value`.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

# All attributes the stdlib puts on a LogRecord by default. Anything *else* on
# the record came from a caller's `extra={...}` and should be rendered.
_RESERVED = set(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
) | {"message", "asctime", "taskName"}

_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# uvicorn puts its own handlers on the "uvicorn" logger and sets propagate=False
# there, so a handler on the ROOT logger never sees an access line. We attach
# directly to the two loggers uvicorn actually writes to.
#
# Deliberately NOT the parent "uvicorn": `uvicorn.error` propagates up to it, so
# a handler on both would be invoked twice for every startup/error line. uvicorn
# only ever logs to these two; the parent exists purely as a config anchor.
_UVICORN_LOGGERS = ("uvicorn.access", "uvicorn.error")

# The single file handler, kept so late attachers (the lifespan re-attach) reuse
# it instead of opening a second handle on the same file.
_file_handler: logging.Handler | None = None


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


def configure_logging(
    level: str = "INFO",
    stream=sys.stderr,
    settings=None,
) -> None:
    """Install the ExtraFormatter on the root logger, plus the rotating file log.

    `settings` is optional so the old two-arg call still works (tests, CLI). Pass
    it to get `logs/app.log` and the JSONL ledger; without it this behaves
    exactly as before — stderr only.

    Safe to call once at process start; re-attaching is idempotent.
    """
    global _file_handler

    log_level = getattr(logging, str(level).upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(log_level)

    formatter = ExtraFormatter(_FORMAT)
    if root.handlers:
        for handler in root.handlers:
            handler.setFormatter(formatter)
            handler.setLevel(log_level)
    elif settings is None or getattr(settings, "log_to_stderr", True):
        handler = logging.StreamHandler(stream)
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        root.addHandler(handler)

    if settings is None:
        return

    log_dir = Path(settings.log_dir)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        _file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "app.log",
            maxBytes=int(settings.app_log_max_bytes),
            backupCount=int(settings.app_log_backup_count),
            encoding="utf-8",
            delay=True,
        )
        _file_handler.setFormatter(formatter)
        _file_handler.setLevel(log_level)
        root.addHandler(_file_handler)
    except OSError as e:
        # A missing/unwritable log dir must not stop the service from starting.
        logging.getLogger(__name__).warning(
            "app_log_file_unavailable", extra={"dir": str(log_dir), "error": str(e)}
        )
        _file_handler = None

    attach_uvicorn_handlers()

    # The JSONL ledger shares the same directory and the same on/off moment.
    import usage_log

    usage_log.configure(settings)


def attach_uvicorn_handlers() -> None:
    """Route uvicorn's own loggers into app.log.

    Called twice on purpose: once at import (uvicorn's dictConfig normally runs
    before the app module is imported) and once from the FastAPI lifespan, which
    covers the embedded `uvicorn.run()` path where the order is reversed and our
    handler would otherwise be installed before uvicorn replaces its own.
    Idempotent — the same handler object is never added twice.
    """
    if _file_handler is None:
        return
    for name in _UVICORN_LOGGERS:
        lg = logging.getLogger(name)
        if _file_handler not in lg.handlers:
            lg.addHandler(_file_handler)
