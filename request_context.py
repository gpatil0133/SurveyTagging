"""The caller's bearer token, scoped to the request that carried it.

The browser sends the platform's `access_token` (read from localStorage by
static/app.js) on every call. Anything this process does *outbound* on that
caller's behalf — apismx today, any other SoGo service later — should travel
with that same token rather than with a shared service credential: same issuer,
same user, correct audit trail.

Passing it down explicitly stops being practical past the first call site, so
it rides a ContextVar instead:

    run.py        middleware sets it from the Authorization header
    smx_runner    build_client() falls back to it when no token was passed

A ContextVar is the right carrier rather than a thread-local because the work
is async and `asyncio.to_thread` copies the current context into the worker —
so a route that offloads blocking share/HTTP work keeps the token, which a
thread-local would lose.

Two places it does NOT reach, both deliberate:
  - `pipeline/orchestrator.py`'s ThreadPoolExecutor workers, which are handed
    raw callables and do not copy context.
  - `scheduler.py`, which has no inbound request at all.
Both are headless paths, and both fall back to `settings.smx_token`.
"""

from __future__ import annotations

from contextvars import ContextVar, Token

_bearer: ContextVar[str] = ContextVar("request_bearer_token", default="")


def set_token(token: str) -> Token:
    """Bind a token to the current context. Returns the reset handle."""
    return _bearer.set(token or "")


def reset_token(handle: Token) -> None:
    """Unbind, restoring whatever was in scope before `set_token`."""
    _bearer.reset(handle)


def current_token() -> str:
    """The inbound caller's raw JWT, or "" outside a request."""
    return _bearer.get()
