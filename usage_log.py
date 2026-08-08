"""Per-unit-of-work usage ledger: cost, tokens, timing, request correlation.

Two sinks, both under `settings.log_dir`:

  app.log                   human/ops text log — see log_config.py
  usage-YYYY-MM-DD.jsonl    this module — one JSON object per line

The JSONL is the billing/perf ledger. Every line carries a `kind` and a
`request_id`, and those two together are the whole design:

    kind="api_request"   one per inbound HTTP call — method, path, status, ms
    kind="survey"        one per survey actually tagged — LLM calls + cost
    kind="tenant"        one per tenant-level unit (tenant_tags + canon builds)

`POST /api/tenants/75885/tag-surveys` therefore emits ONE `api_request` line and
N `survey` lines plus a `tenant` line, all sharing a single `request_id`. Cost
per request is `SUM(cost_usd) WHERE request_id = ?`; cost per survey is the
`survey` line itself. Nothing needs to be reconstructed from text logs.

Why a ContextVar rather than threading a collector through every signature:
the call that spends the money (`LLMClient.complete`) sits four frames below the
code that knows which survey is being tagged, behind a `loop.run_until_complete`
that would otherwise have to grow a parameter at every level. A ContextVar set
on the tagging thread is visible inside that loop's task, because asyncio copies
the current context when it creates the task.

The one place context does NOT flow for free is the orchestrator's
ThreadPoolExecutor, which hands workers raw callables — exactly the gap
documented in request_context.py. `snapshot()` / `restore()` close it: the
parent captures before fanning out, each worker re-binds on entry.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger("survey_tagging.usage")

# ---------------------------------------------------------------- config ----

_enabled = False
_log_dir: Path | None = None
_price_in_per_mtok = 0.0
_price_out_per_mtok = 0.0
_write_lock = threading.Lock()


def configure(settings) -> None:
    """Wire the ledger from Settings. Called once from log_config.configure_logging."""
    global _enabled, _log_dir, _price_in_per_mtok, _price_out_per_mtok
    _log_dir = Path(settings.log_dir)
    _enabled = bool(settings.usage_log_enabled)
    _price_in_per_mtok = float(getattr(settings, "llm_price_input_per_mtok", 0.0) or 0.0)
    _price_out_per_mtok = float(getattr(settings, "llm_price_output_per_mtok", 0.0) or 0.0)
    if _enabled:
        _log_dir.mkdir(parents=True, exist_ok=True)


def ledger_path(day: date | None = None) -> Path:
    """Today's JSONL file. Daily rotation keeps a day's spend greppable as a unit."""
    root = _log_dir or Path("./logs")
    return root / f"usage-{(day or datetime.now(timezone.utc).date()).isoformat()}.jsonl"


# --------------------------------------------------------------- context ----

_request_id: ContextVar[str] = ContextVar("usage_request_id", default="")
_scope: ContextVar["Scope | None"] = ContextVar("usage_scope", default=None)


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def bind_request(request_id: str) -> Token:
    return _request_id.set(request_id or "")


def reset_request(handle: Token) -> None:
    _request_id.reset(handle)


def current_request_id() -> str:
    """The in-flight request's id, or "" on headless paths (scheduler, CLI)."""
    return _request_id.get()


def snapshot() -> dict:
    """Capture the ambient context so a worker thread can re-enter it.

    Call in the PARENT thread before fanning out. The active scope is
    deliberately NOT carried: a worker opens its own per-survey scope, and
    inheriting the parent's would attribute every worker's tokens to one record.
    """
    return {"request_id": _request_id.get()}


def restore(snap: dict) -> None:
    """Re-bind a `snapshot()` inside a worker thread. Not reset — the thread's
    context dies with the task."""
    _request_id.set(snap.get("request_id", ""))


# ----------------------------------------------------------------- scopes ----


class Scope:
    """One accumulating unit of work. Built by `scope()`; not constructed directly."""

    __slots__ = ("kind", "tenant_id", "survey_no", "fields", "started",
                 "calls", "status", "error")

    def __init__(self, kind: str, tenant_id: int | None,
                 survey_no: int | None, fields: dict) -> None:
        self.kind = kind
        self.tenant_id = tenant_id
        self.survey_no = survey_no
        self.fields = dict(fields)
        self.started = time.perf_counter()
        self.calls: list[dict] = []
        self.status = "success"
        self.error: str | None = None

    def totals(self) -> dict:
        """Roll the individual LLM calls up into the per-record `llm` block.

        `cached_calls` are disk-cache replays: they cost nothing and took no
        tokens, so they are counted but contribute zero. A survey whose calls
        are all cached shows `calls=2, cached_calls=2, cost_usd=0.0` — which is
        the honest answer to "what did this re-tag cost".
        """
        agg = {
            "calls": len(self.calls),
            "cached_calls": sum(1 for c in self.calls if c.get("cached")),
            "failed_calls": sum(1 for c in self.calls if not c.get("ok", True)),
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "cost_usd": 0.0,
            "llm_ms": 0,
        }
        for c in self.calls:
            agg["input_tokens"] += c.get("input_tokens") or 0
            agg["output_tokens"] += c.get("output_tokens") or 0
            agg["cache_read_tokens"] += c.get("cache_read_tokens") or 0
            agg["cache_write_tokens"] += c.get("cache_write_tokens") or 0
            agg["cost_usd"] += c.get("cost_usd") or 0.0
            agg["llm_ms"] += c.get("duration_ms") or 0
        agg["cost_usd"] = round(agg["cost_usd"], 6)
        return agg


@contextmanager
def scope(
    kind: str,
    *,
    tenant_id: int | None = None,
    survey_no: int | None = None,
    **fields: Any,
) -> Iterator[Scope]:
    """Open a unit of work; emit one JSONL record when it closes.

    Always emits — including on the exception path, where the record carries
    `status="failed"` and the error. A failed survey still burned tokens, and a
    ledger that only records successes understates the bill.
    """
    sc = Scope(kind, tenant_id, survey_no, fields)
    handle = _scope.set(sc)
    try:
        yield sc
    except BaseException as e:
        sc.status = "failed"
        sc.error = f"{type(e).__name__}: {e}"
        raise
    finally:
        _scope.reset(handle)
        _emit(sc)


def annotate(**fields: Any) -> None:
    """Attach fields to the active scope from anywhere below it. No-op outside one."""
    sc = _scope.get()
    if sc is not None:
        sc.fields.update(fields)


def set_status(status: str) -> None:
    """Override the scope's outcome (e.g. "skipped" for an unchanged survey)."""
    sc = _scope.get()
    if sc is not None:
        sc.status = status


# ------------------------------------------------------------- LLM sink ----


def record_llm_call(
    *,
    call_type: str,
    model: str,
    cached: bool = False,
    ok: bool = True,
    duration_ms: int = 0,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cache_read_tokens: int | None = None,
    cache_write_tokens: int | None = None,
    cost_usd: float | None = None,
    attempt: int = 1,
    error: str | None = None,
) -> None:
    """Record one provider round-trip against the active scope.

    Called from LLMClient. No-op when no scope is open (ad-hoc `/api/tag`,
    tests), so the client stays usable outside the pipeline.
    """
    sc = _scope.get()
    if sc is None:
        return
    entry = {
        "call_type": call_type,
        "model": model,
        "cached": cached,
        "ok": ok,
        "duration_ms": duration_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
        "cost_usd": cost_usd,
        "attempt": attempt,
    }
    if error:
        entry["error"] = error
    sc.calls.append(entry)


def estimate_cost(
    response: Any,
    input_tokens: int,
    output_tokens: int,
) -> float | None:
    """Cost for one call, in USD.

    litellm's price map is the source of truth — it covers Anthropic and OpenAI
    ids alike, so this survives a provider switch untouched. It returns 0.0 (or
    raises) for a model it does not know, which is indistinguishable from a
    genuinely free call; the configured per-Mtok override is the escape hatch
    for that case. Returns None when neither can price it, so the ledger stores
    an explicit null rather than a fake zero.
    """
    try:
        import litellm

        cost = litellm.completion_cost(completion_response=response)
        if cost:
            return float(cost)
    except Exception as e:  # noqa: BLE001 — pricing must never break a tag run
        logger.debug("cost_lookup_failed", extra={"error": str(e)})

    if _price_in_per_mtok or _price_out_per_mtok:
        return round(
            (input_tokens / 1_000_000) * _price_in_per_mtok
            + (output_tokens / 1_000_000) * _price_out_per_mtok,
            6,
        )
    return None


# ------------------------------------------------------------- emission ----


def _emit(sc: Scope) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "kind": sc.kind,
        "request_id": current_request_id(),
        "tenant_id": sc.tenant_id,
        "survey_no": sc.survey_no,
        "status": sc.status,
        "duration_ms": int((time.perf_counter() - sc.started) * 1000),
        "pid": os.getpid(),
        "thread": threading.current_thread().name,
    }
    if sc.error:
        record["error"] = sc.error
    record.update(sc.fields)
    if sc.calls:
        record["llm"] = sc.totals()
        record["llm_calls"] = sc.calls
    write(record)


def write(record: dict) -> None:
    """Append one record to today's ledger.

    Serialized on a lock and opened per write: appends of a single short line
    are atomic enough on both NTFS and POSIX, and holding a handle open across a
    day boundary would keep writing to yesterday's file. Failures here are
    swallowed to a WARNING — a full disk must not fail a tagging run.
    """
    if not _enabled:
        return
    try:
        line = json.dumps(record, ensure_ascii=False, default=str)
        path = ledger_path()
        with _write_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception as e:  # noqa: BLE001
        logger.warning("usage_log_write_failed", extra={"error": str(e)})
