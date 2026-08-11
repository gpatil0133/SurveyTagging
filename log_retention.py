"""Age-based retention for the two log sinks under `settings.log_dir`.

Rotation and retention answer different questions and both are needed:

  rotation   how large may ONE file get before it is rolled aside?
             app.log — RotatingFileHandler, `app_log_max_bytes`
             usage   — one file per UTC day, rolled by filename
  retention  how long does the resulting PILE live?
             this module

Without retention `app.log.N` is capped by `app_log_backup_count`, but the
ledger is not capped at all: `usage-YYYY-MM-DD.jsonl` is a new file every day
and nothing ever removed one, so a long-lived service filled its log volume one
day at a time. Both sinks are pruned here so there is a single place that
deletes a log file.

Two rules exist because the two sinks date themselves differently:

  app.log.*   pruned by MTIME. RotatingFileHandler renames on rollover, and a
              rename preserves mtime, so `app.log.7`'s mtime is still the moment
              its last line was written. Gaps left by pruning are harmless —
              `doRollover` only renames backups that exist.
  usage-*     pruned by the DATE IN THE NAME, which survives a copy/restore that
  smx-wire-*  would reset mtime and make a month-old ledger look new. Falls back
              to mtime only when the name does not parse.

The wire trace (smx-wire-*.jsonl, tenant_profile/smx_trace.py) is dated the same
way but is a DEBUG artifact rather than a record, so it expires on the app.log
cutoff, not the ledger's. Nobody bills from it and it is written with full
request/response bodies — keeping it as long as the billing ledger would be the
wrong default in both size and sensitivity.

The live `app.log` and today's ledger are never deleted, whatever the cutoff.
`sweep()` never raises: a locked or vanished file (both routine on Windows, where
another process may hold a handle) skips that file and the rest continue.
"""

from __future__ import annotations

import logging
import re
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

APP_LOG_NAME = "app.log"
_USAGE_DATED = re.compile(r"^usage-(\d{4})-(\d{2})-(\d{2})\.jsonl$")
_WIRE_DATED = re.compile(r"^smx-wire-(\d{4})-(\d{2})-(\d{2})\.jsonl$")

# Configured once from log_config.configure_logging; unset means "no retention",
# which is what the CLI and the tests that call configure_logging(level) get.
_log_dir: Path | None = None
_app_days = 0
_usage_days = 0

_lock = threading.Lock()
_last_swept: date | None = None


def configure(settings) -> None:
    """Wire retention from Settings. Called from log_config.configure_logging."""
    global _log_dir, _app_days, _usage_days, _last_swept
    _log_dir = Path(settings.log_dir)
    _app_days = int(getattr(settings, "app_log_retention_days", 0) or 0)
    _usage_days = int(getattr(settings, "usage_log_retention_days", 0) or 0)
    _last_swept = None


def enabled() -> bool:
    return _log_dir is not None and (_app_days > 0 or _usage_days > 0)


def sweep_now() -> int:
    """Prune with the configured settings. Returns the number of files deleted."""
    if not enabled():
        return 0
    assert _log_dir is not None
    return sweep(_log_dir, app_days=_app_days, usage_days=_usage_days)


def maybe_sweep() -> int:
    """Sweep at most once per UTC day.

    Called from the ledger's write path, so the guard is checked before the lock
    is taken — every line written pays a date comparison, not a lock acquisition.
    """
    if not enabled():
        return 0
    today = datetime.now(timezone.utc).date()
    if _last_swept == today:
        return 0
    with _lock:
        if _last_swept == today:  # another thread crossed the day boundary first
            return 0
        _mark_swept(today)
    return sweep_now()


def _mark_swept(day: date) -> None:
    global _last_swept
    _last_swept = day


def sweep(
    log_dir: Path | str,
    *,
    app_days: int = 0,
    usage_days: int = 0,
    now: datetime | None = None,
) -> int:
    """Delete rotated app logs and dated ledgers older than their cutoffs.

    `app_days` / `usage_days` of 0 (or less) disable that sink's sweep — nothing
    is deleted rather than everything, which is the safe reading of "unset".
    Pure function of its arguments so tests need no module state.
    """
    root = Path(log_dir)
    stamp = now or datetime.now(timezone.utc)
    deleted: list[str] = []

    try:
        entries = list(root.iterdir())
    except OSError as e:
        logger.debug("log_retention_scan_failed", extra={"dir": str(root), "error": str(e)})
        return 0

    for path in entries:
        expiry = _expiry_days(path.name, app_days, usage_days)
        if expiry is None:
            continue
        age = _age_days(path, stamp)
        if age is None or age <= expiry:
            continue
        try:
            path.unlink()
            deleted.append(path.name)
        except OSError as e:
            # Held open by another process (or already gone). Next sweep retries.
            logger.debug(
                "log_retention_delete_failed",
                extra={"file": path.name, "error": str(e)},
            )

    if deleted:
        logger.info(
            "log_retention_pruned",
            extra={"count": len(deleted), "files": ", ".join(sorted(deleted))},
        )
    return len(deleted)


def _expiry_days(name: str, app_days: int, usage_days: int) -> int | None:
    """Retention for one filename, or None if this file is not ours to delete."""
    if name == APP_LOG_NAME:
        return None  # the live handle — rotation, not retention, bounds it
    if app_days > 0 and name.startswith(APP_LOG_NAME + "."):
        return app_days
    if usage_days > 0 and _USAGE_DATED.match(name):
        return usage_days
    # The wire trace is a debug sink, so it lives on the app.log cutoff. Today's
    # file is deleted only once it is a day past that — unlike app.log there is
    # no open handle to protect, and `_append` reopens per line anyway.
    if app_days > 0 and _WIRE_DATED.match(name):
        return app_days
    return None


def _age_days(path: Path, now: datetime) -> float | None:
    """Age in days: from the date in the name for dated JSONL, else from mtime."""
    match = _USAGE_DATED.match(path.name) or _WIRE_DATED.match(path.name)
    if match:
        try:
            named = date(int(match[1]), int(match[2]), int(match[3]))
        except ValueError:  # e.g. usage-2026-02-31.jsonl — not a real day
            named = None
        if named is not None:
            return (now.date() - named).days
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    return (now - datetime.fromtimestamp(mtime, timezone.utc)) / timedelta(days=1)
