"""Shared request-layer dependencies: the process context and the 500 helper.

`ctx` is built exactly once, by `run.py`, and handed here — the routers import it
from this module rather than reaching back into `run`, which would be a cycle
(`run` imports the routers). Everything a handler needs about the process lives on
it: settings, taxonomy, tagger registry, change detector, LLM client.
"""

from __future__ import annotations

from fastapi import HTTPException

import usage_log
from bootstrap import AppContext

# Populated by `run.configure()` before any router is included. Not Optional in
# practice: importing a router without the app having been built is a programming
# error, and failing on attribute access at that point is the clearest report.
ctx: AppContext = None          # type: ignore[assignment]
settings = None                 # type: ignore[assignment]


def configure(app_context: AppContext) -> None:
    """Hand the routers the one AppContext. Called from run.py at import time."""
    global ctx, settings
    ctx = app_context
    settings = app_context.settings


def server_error(detail: str) -> HTTPException:
    """A 500 that names the request instead of the server's internals.

    The tag routes used to interpolate the caught exception straight into the
    response body, which puts share paths, UNC hosts and driver messages in front
    of whoever called the API. The exception is already logged in full with the
    request id by the middleware; the caller gets the id and nothing else.
    """
    request_id = usage_log.current_request_id()
    suffix = f" (request_id={request_id})" if request_id else ""
    return HTTPException(500, f"{detail}. See server logs{suffix}.")
