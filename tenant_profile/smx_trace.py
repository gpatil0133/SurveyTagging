"""Wire-level trace for the apismx / apipmx exchange.

Everything this service knows about a tenant's profile moves over four calls
(`/AIAccountProfile/Details`, `/AIAccountProfile/Generate`, and apipmx
`/ecdata` / `/dcdata` — the platform encrypts in both directions), and until
this module existed none of them left a usable record. httpx logs a
one-line `HTTP Request: POST ... 500 Internal Server Error` and `SmxClient`
truncated the server's explanation to 200 characters into an exception that only
ever reached the browser. So the log said a call failed and nothing about why —
or about what the successful calls actually returned.

Two sinks, deliberately different in kind (the same split as log_config.py):

  app.log                    one line per step of the exchange, bodies clipped
                             to `max_chars`. For reading while something is
                             wrong.
  smx-wire-YYYY-MM-DD.jsonl  one JSON object per step, bodies NOT clipped (bar a
                             runaway guard). For answering "what exactly did the
                             service send us" after the fact — jq, not eyeballs.

Levels are the whole ergonomics of this thing. The service already logs at INFO
and its INFO stream is dominated by smbprotocol, so a trace at DEBUG would be
invisible without turning on a level that buries it. Hence:

  trace disabled (default)   steps log at DEBUG — off in practice, zero cost
  trace enabled              steps log at INFO, and the JSONL is written
  errors                     ALWAYS log at ERROR with the full body, enabled or
                             not. A 500 you have to reproduce with a flag flipped
                             is a 500 you have already lost.

`SURVEY_TAGGER_SMX_DEBUG_WIRE=true` is the flag; see `smx_runner.build_client`,
which is the one place an `SmxClient` is constructed from Settings.

The bearer token is never passed in here. It lives on the httpx client's default
headers, and nothing in this module reads them — the trace records the URL, the
params and the body, so a JSONL file is shareable without leaking a credential.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("survey_tagging.smx.wire")

# Everything the stdlib puts on a LogRecord itself. An `extra={...}` key that
# collides with one of these makes logging raise ("Attempt to overwrite ..."),
# which would turn a diagnostic into an outage; `_log_fields` renames instead.
_RESERVED = set(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
) | {"message", "asctime", "taskName"}

# Runaway guard for the JSONL. Far above any real profile payload (~50 KB), low
# enough that a pathological response cannot write a gigabyte line.
_JSONL_MAX_CHARS = 200_000

_counter = itertools.count(1)
_write_lock = threading.Lock()


def compact(value: Any) -> str:
    """One-line JSON for a body, falling back to repr for anything unserializable."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(value)


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...(+{len(text) - limit} more chars)"


class SmxTrace:
    """Records the steps of one apismx exchange. Cheap and inert when disabled.

    Construct one per `SmxClient` (i.e. per request or per headless run). The
    exchange counter is process-wide so a line in app.log and a record in the
    JSONL can be tied together across concurrent tenants.
    """

    __slots__ = ("enabled", "log_dir", "max_chars")

    def __init__(
        self,
        *,
        enabled: bool = False,
        log_dir: Path | str | None = None,
        max_chars: int = 2000,
    ) -> None:
        self.enabled = bool(enabled)
        self.log_dir = Path(log_dir) if log_dir else None
        # 200 chars is what the old truncation gave us and it was not enough to
        # read an ASP.NET exception; refuse to be configured back down to that.
        self.max_chars = max(200, int(max_chars or 2000))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"SmxTrace(enabled={self.enabled}, log_dir={self.log_dir}, "
                f"max_chars={self.max_chars})")

    def new_exchange(self) -> str:
        """An id shared by every step of one request/response/decrypt cycle."""
        return f"x{next(_counter):05d}"

    def record(self, event: str, *, level: int | None = None, **fields: Any) -> None:
        """Emit one step. `level=None` means "trace level" (DEBUG, or INFO when on).

        Pass `level=logging.ERROR` for anything a reader needs without having
        turned the flag on first.
        """
        effective = level if level is not None else (
            logging.INFO if self.enabled else logging.DEBUG
        )
        if logger.isEnabledFor(effective):
            logger.log(effective, event, extra=self._log_fields(fields))
        if self.enabled and self.log_dir is not None:
            self._append(event, fields)

    # ------------------------------------------------------------- internals --

    def _log_fields(self, fields: dict[str, Any]) -> dict[str, Any]:
        """Clip bodies and dodge LogRecord's reserved attribute names."""
        out: dict[str, Any] = {}
        for key, value in fields.items():
            if value is None:
                continue
            safe = f"{key}_" if key in _RESERVED else key
            if isinstance(value, (str, dict, list)):
                out[safe] = _clip(compact(value), self.max_chars)
            else:
                out[safe] = value
        return out

    def wire_path(self, day: Any = None) -> Path:
        """Today's wire file. Daily, like the usage ledger, and swept by the same
        retention pass (log_retention treats it as a debug log, not a record)."""
        root = self.log_dir or Path("./logs")
        stamp = day or datetime.now(timezone.utc).date()
        return root / f"smx-wire-{stamp.isoformat()}.jsonl"

    def _append(self, event: str, fields: dict[str, Any]) -> None:
        """Append the full record. Never raises — a trace must not break a fetch."""
        try:
            import usage_log

            request_id = usage_log.current_request_id()
        except Exception:  # noqa: BLE001 - correlation is a nicety, not a requirement
            request_id = ""

        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event": event,
            "request_id": request_id,
            "pid": os.getpid(),
            "thread": threading.current_thread().name,
        }
        for key, value in fields.items():
            if value is None:
                continue
            record[key] = (_clip(value, _JSONL_MAX_CHARS)
                           if isinstance(value, str) else value)

        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
            path = self.wire_path()
            with _write_lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except Exception as e:  # noqa: BLE001
            logger.warning("smx_wire_write_failed", extra={"error": str(e)})


# A shared no-op instance for the default path, so `SmxClient` never has to
# branch on `if self._trace is not None`.
DISABLED = SmxTrace()
