"""Standalone FastAPI service for the Survey Auto-Tagger.

The single entry point. All wiring comes from `bootstrap.build_context()`; all
tagging work goes through `service.py` (which drives the per-survey engine for a
single survey and the orchestrator for tenant-level work).

This module owns the app itself — middleware, lifespan, the static UI — and the
ORDER the routers are registered in. The handlers live in `api/`, one module per
resource group (§1 surveys, §2 tenant_surveys, §3 tenant_tags, §4 profile,
§5 catalog, §6 admin); they read the process context from `api/deps.py`.

Surface (all under /api). The sections below are the order the routers are
included in; see "Route layout" further down for why that order is what it is.

  §1  Survey  — one survey, addressed by tenant + survey number
    POST   /tenants/{t}/surveys/{s}/tag        tag one survey (incremental)
    POST   /tenants/{t}/surveys/{s}/retag      force re-tag one survey
    GET    /tenants/{t}/surveys/{s}/tags       unified survey view (ETag/304)
    DELETE /tenants/{t}/surveys/{s}/tags       delete one survey's tags
    POST   /tag                                ad-hoc: tag uploaded survey JSON
  §2  Tenant surveys  — the same work fanned out over every survey
    POST   /tenants/{t}/tag-surveys            tag all surveys (bounded-parallel)
    POST   /tenants/{t}/retag-surveys          force re-tag all surveys
    GET    /tenants/{t}/tag-surveys            tagged-status for the tenant
    GET    /tenants/{t}/tag-surveys/stream     same, NDJSON, one line per survey
  §3  Tenant tags  — tenant_tags.json
    POST   /tenants/{t}/tag                    build tenant_tags.json
    GET    /tenants/{t}/tags                   read tenant_tags.json
    DELETE /tenants/{t}/tags                   delete tenant_tags.json
  §4  Tenant profile  — org/cx/ex artifacts (producer set by `profile_source`)
    POST   /tenants/{t}/profile/fetch          resolve: share -> producer -> generate
    GET    /tenants/{t}/profile                summary of what is on disk
    GET    /tenants/{t}/profile/diagnose       read-only: why is there no profile?
    GET    /tenants/{t}/profile/{agent}        raw envelope for org | cx | ex
    DELETE /tenants/{t}/profile
  §5  Catalog  — process-wide, read-only, no tenant
    GET    /taxonomy               all 50 dimensions + the explanation layer
    GET    /config                 server config the UI shapes itself from
    GET    /me                     caller identity from the Bearer token
    GET    /health/share           is the data root reachable?
    GET    /surveys                whole-root catalog (expensive; legacy callers)
  §6  Admin
    GET    /admin/autoretag        (see scheduler.py)
    POST   /admin/autoretag/run-now
  §7  Static UI
    GET    /                       index.html; /static/* is a mount

Every `/tenants/{t}/…` route above is also registered without the `/tenants/{t}`
prefix (`POST /api/surveys/{s}/tag`, `GET /api/tags`, `GET /api/profile`, …).
On those forms the tenant comes from the caller's JWT — see the note above
`capture_bearer_token`. The one that is not a straight prefix-drop is
`POST /api/tenants/{t}/tag`, whose short form is `POST /api/tags`, because
`POST /api/tag` already means "tag this uploaded survey JSON".

Nothing here is authenticated: `auth_enabled` is False by default and the token
is read for tenant resolution and outbound forwarding only.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure the package modules are importable when running from this folder.
sys.path.insert(0, str(Path(__file__).parent))

# Load .env into os.environ so libraries that read env vars directly
# (litellm reads ANTHROPIC_API_KEY; the parallel-web SDK reads PARALLEL_API_KEY)
# see the same values Pydantic Settings does.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass

from contextlib import asynccontextmanager

from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import auth
import request_context
from bootstrap import build_context

import usage_log
from log_config import attach_uvicorn_handlers, configure_logging
from settings import Settings

_boot_settings = Settings()
configure_logging(_boot_settings.log_level, settings=_boot_settings)
logger = logging.getLogger("survey_tagging.api")

# Single composition root for the whole process. The Settings built above for
# logging is handed in rather than letting bootstrap construct a second one.
_ctx = build_context(_boot_settings)
_settings = _ctx.settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Re-assert the file handler on uvicorn's loggers. When the server is
    # started as `python run.py`, uvicorn installs its own logging config
    # *after* this module was imported and would otherwise leave app.log with
    # no access lines in it.
    attach_uvicorn_handlers()

    # V9 removed the embedding warm-up that used to run here on a daemon
    # thread. `import sentence_transformers` cost ~98s on a cold process (plus
    # ~6s to construct the model and take a first forward pass), which is how a
    # single 7-question survey once took 143 seconds with 66% of it inside a
    # Python import. The journey leaves are now inlined into the question prompt
    # instead of being ranked by an encoder, so there is no model to load and
    # startup carries no warm-up cost at all.

    # Start the env-gated periodic auto-retag scheduler (OFF by default).
    from scheduler import AutoRetagScheduler
    app.state.scheduler = AutoRetagScheduler(_ctx)
    await app.state.scheduler.start()
    try:
        yield
    finally:
        await app.state.scheduler.stop()


# Swagger is hand-rolled below rather than configured through `root_path`.
#
# The obvious move behind a sub-path is FastAPI(root_path=path_prefix), and it
# is wrong here. ASGI's contract is that `root_path` is a prefix the incoming
# `path` still CARRIES; our proxy strips it (web.config rewrites
# /apisurveytagging/api/x to /api/x), so the two disagree. Top-level routes
# survive that — Starlette's `get_route_path` leaves a path alone when it does
# not start with the root_path — but a Mount does not: it appends its own
# segment to root_path, gets "/apisurveytagging/static" versus a path of
# "/static/app.css", takes the same escape hatch, and hands StaticFiles the
# UNSTRIPPED path. Every asset under /static then 404s. Measured, not theorised.
#
# So the prefix is applied only where it is actually needed — the `servers[]`
# entry that makes Swagger's "Try it out" hit the public URL — and the docs
# routes are declared with a relative openapi_url so they work at the origin
# root and under a virtual path with no configuration at all.
app = FastAPI(
    title="Survey Auto-Tagger",
    version="7.0",
    lifespan=lifespan,
    docs_url=None,
    openapi_url=None,
    redoc_url=None,
)


@app.get("/openapi.json", include_in_schema=False)
async def openapi_spec(request: Request) -> dict:
    """The schema, with `servers[]` pointing at the app's public base.

    `path_prefix` is the answer when it is configured. When it is not, the
    request itself is asked: ARR forwards X-Forwarded-Prefix, and failing that
    the Referer of the /docs page that fetched this carries the prefix in its
    own path. Both are guesses, which is why the setting wins.
    """
    spec = dict(app.openapi())
    prefix = _settings.path_prefix
    if not prefix:
        prefix = request.headers.get("x-forwarded-prefix", "").rstrip("/")
    if not prefix:
        referer_path = urlparse(request.headers.get("referer", "")).path.rstrip("/")
        if referer_path.endswith("/docs"):
            prefix = referer_path[: -len("/docs")]
    if prefix:
        spec["servers"] = [{"url": prefix}]
    return spec


@app.get("/docs", include_in_schema=False)
async def swagger_ui() -> HTMLResponse:
    """Swagger UI. `openapi_url` is RELATIVE on purpose — from /docs it resolves
    to /openapi.json and from /apisurveytagging/docs to
    /apisurveytagging/openapi.json, with nothing to configure either way.

    Note this page loads swagger-ui's assets from a CDN, so on the air-gapped
    deployment network it renders blank. The spec at /openapi.json is the part
    that works everywhere.
    """
    return get_swagger_ui_html(
        openapi_url="openapi.json", title="Survey Auto-Tagger — Swagger UI"
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def track_request(request: Request, call_next):
    """Give every request an id, and write one `api_request` ledger line for it.

    The id is the join key for the whole ledger: a `POST /tag-surveys` that tags
    40 surveys produces 1 `api_request` record, 40 `survey` records and 1
    `tenant` record, all carrying it. `SUM(llm.cost_usd) GROUP BY request_id`
    is then the cost of an API call, and the same query grouped by `survey_no`
    is the cost of a survey.

    An inbound `X-Request-ID` is honoured so the platform's own correlation id
    wins when there is one; the id is echoed back on the response either way.

    Only `/api/*` is recorded. Static asset hits would swamp the ledger and are
    already visible in uvicorn's access log in app.log.
    """
    request_id = request.headers.get("x-request-id") or usage_log.new_request_id()
    handle = usage_log.bind_request(request_id)
    tracked = request.url.path.startswith("/api/")
    started = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        if tracked:
            usage_log.write({
                "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "kind": "api_request",
                "request_id": request_id,
                "method": request.method,
                # Group latency by handler, not by concrete path — otherwise
                # every tenant id splinters into its own bucket. The handler
                # name also merges the two URL forms of each route
                # (/api/tenants/{t}/surveys/{s}/tag and /api/surveys/{s}/tag are
                # one endpoint). Null on a 404, which matches no route.
                "endpoint": _endpoint_name(request),
                "path": request.url.path,
                "status_code": status_code,
                "duration_ms": int((time.perf_counter() - started) * 1000),
                "client": request.client.host if request.client else None,
            })
        usage_log.reset_request(handle)


def _endpoint_name(request: Request) -> str | None:
    """Handler name for the matched route.

    Starlette 0.27 puts `endpoint` (the function) on the scope during routing
    but not `route`, so there is no path template to read — the function's name
    is the stable grouping key available here.
    """
    endpoint = request.scope.get("endpoint")
    return getattr(endpoint, "__name__", None)


@app.middleware("http")
async def capture_bearer_token(request: Request, call_next):
    """Put the caller's JWT in the request context for the life of the request.

    The UI sends one on every call — the platform's `access_token` out of
    localStorage, or a token pasted into its Tenant Profile panel, which wins
    over it; this is what lets anything downstream (apismx today) forward
    that same token outbound without every intermediate signature having to
    carry it. Absent header → empty string → callers fall back to
    SURVEY_TAGGER_SMX_TOKEN, exactly as before.

    This does not authenticate anything. Enforcement is still `auth.require_auth`
    behind `SURVEY_TAGGER_AUTH_ENABLED`.
    """
    handle = request_context.set_token(auth.bearer_token(request.headers.get("authorization")))
    try:
        return await call_next(request)
    finally:
        request_context.reset_token(handle)


class PathPrefixMiddleware:
    """Accept the app's own public prefix in the request path.

    `path_prefix` exists so the app can EMIT correct URLs behind an IIS virtual
    application — the UI shell bakes it into `ST_BASE_PATH`, so the browser then
    asks for `/apisurveytagging/static/app.css` and `/apisurveytagging/api/...`.
    On the deployed box ARR strips that prefix before uvicorn ever sees it, so
    the app only has to match `/static/...` and `/api/...`.

    Run the same app directly — `python run.py`, no IIS — and there is nothing in
    front to do the stripping, so every asset and every API call arrives with a
    prefix no route matches. The UI comes up blank against a server that is
    working perfectly. That is the entire local/IIS incompatibility, and this is
    the whole of the fix: strip the prefix here too, so the app answers on BOTH
    URL shapes and one `.env` serves both ways of running it.

    Safe in front of a proxy that already stripped it: a path that does not
    carry the prefix is passed through untouched, which is the case on every
    deployed request. Same `_strip_prefix` contract as `wsgi_app.py`, which has
    to do this for the FastCGI path for the same reason.

    Pure ASGI rather than `@app.middleware("http")` because it must also cover
    WebSocket scopes and, more importantly, run OUTSIDE `track_request` — the
    ledger's `path` field should read the same on a local request as on a
    deployed one, and it only does if the prefix is gone before that middleware
    sees it.
    """

    def __init__(self, app, prefix: str) -> None:
        self.app = app
        self.prefix = prefix

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            path = scope.get("path", "") or ""
            stripped = _strip_path_prefix(path, self.prefix)
            if stripped != path:
                # Copied, not mutated: the server owns the scope dict.
                scope = dict(scope)
                scope["path"] = stripped
                raw = scope.get("raw_path")
                if raw:
                    # raw_path is bytes and keeps the query string off; strip the
                    # same number of bytes so the two cannot disagree about which
                    # path was requested.
                    scope["raw_path"] = raw[len(self.prefix.encode()):] or b"/"
        await self.app(scope, receive, send)


def _strip_path_prefix(path: str, prefix: str) -> str:
    """The path to route on, with the mount prefix removed.

    Only a whole leading SEGMENT is stripped, so a route that merely starts with
    the same letters is left alone. Returns the path unchanged when there is
    nothing to strip — including when something in front already stripped it.
    """
    if not prefix:
        return path
    if path == prefix:
        return "/"
    if path.startswith(f"{prefix}/"):
        return path[len(prefix):]
    return path


# Added last so it ends up OUTERMOST: Starlette builds the stack in reverse, so
# the final `add_middleware` call is the first to see a request. Registered only
# when there is a prefix — an empty one makes every branch above a no-op, and a
# middleware that can never do anything is worth not having in the stack.
if _settings.path_prefix:
    app.add_middleware(PathPrefixMiddleware, prefix=_settings.path_prefix)


# ====================================================================
# Route layout
# ====================================================================
#
# Four rules decide where a route goes. The first is a correctness constraint;
# the rest are conventions that keep the file readable — but only the first one
# will break the API if you get it wrong.
#
# 1. LITERAL BEFORE PARAMETER — the one rule with teeth.
#    FastAPI matches routes in *definition order* and takes the first path that
#    fits, with no notion of one pattern being more specific than another. So a
#    literal segment must be registered before any `{param}` at the same
#    position that could swallow it:
#
#        GET /profile/diagnose      <- must come first
#        GET /profile/{agent}          otherwise "diagnose" arrives as an agent
#                                      name and the route 400s
#
#    Only same-method pairs collide. `POST /profile/fetch` sits happily above
#    `GET /profile/{agent}` because the methods differ — though note the
#    consequence: `GET /api/profile/fetch` does match `{agent}` and answers
#    "Unknown agent: 'fetch'" rather than 405.
#
# 2. GROUPED BY RESOURCE, NARROWEST FIRST.
#    One survey (§1) -> every survey under a tenant (§2) -> the tenant's own
#    tags (§3) -> the tenant's profile (§4) -> process-wide reads (§5) ->
#    admin (§6) -> the UI (§7). Reading top to bottom widens the blast radius of
#    what a call touches, which is also roughly the order a new tenant is
#    onboarded in: fetch a profile, build tenant tags, tag the surveys.
#
# 3. LIFECYCLE ORDER WITHIN A GROUP: build -> read -> delete
#    (POST, then GET, then DELETE). Where a group has both an incremental and a
#    forced writer the incremental one leads, since it is the common call.
#    §1 breaks the pattern once, deliberately: `POST /tag` is last because it is
#    the only route in the group that has no tenant and touches no disk.
#
# 4. HELPERS ABOVE THE ROUTES THEY SERVE, never interleaved between two
#    handlers. Each section opens with its module-level constants, request
#    models and private `_helpers`, then runs its routes contiguously — so the
#    surface of a section can be read without stepping over implementation.
#
# Decorator stacking is a separate axis: the long `/tenants/{t}/…` form is
# always written on top and the short form beneath it, purely so the canonical
# path is the one you read first. Python applies decorators bottom-up, so the
# short form is in fact registered first — harmless here, because the two never
# overlap. If you ever stack two forms that *can* match the same URL, rule 1
# applies to the effective (bottom-up) order, not the written one.
#
# `tenant_id` is optional on every tenant route below. Each is registered twice:
#
#     /api/tenants/{tenant_id}/surveys/{s}/tags     tenant in the path
#     /api/surveys/{s}/tags                         tenant from the token
#
# On the short form FastAPI reads `tenant_id` as a query param instead, so
# `?tenant_id=` works there too. The URL always wins and is never cross-checked
# against the token: tenants that exist only on the net-share have no platform
# account for a token to agree with. See auth.resolve_tenant_id.



# ====================================================================
# Routers — registration ORDER IS THE MATCHING ORDER (see above)
#
# Narrowest resource first, exactly as the sections used to sit in this file:
# one survey, then all of a tenant's surveys, then the tenant's own tags, then
# its profile, then the process-wide reads, then admin. §7 (the static UI) stays
# below them because `GET /` is the broadest match in the app.
#
# Each router keeps its own section notes; the rules above apply WITHIN a router
# too, which is why `/profile/diagnose` and `/profile/{agent}` had to stay in the
# same module. `tests/test_api/test_route_surface.py` pins both the inventory and
# the two orderings that are correctness rather than style.
# ====================================================================

from api import admin, catalog, deps, profile, surveys, tenant_surveys, tenant_tags  # noqa: E402

deps.configure(_ctx)

app.include_router(surveys.router)
app.include_router(tenant_surveys.router)
app.include_router(tenant_tags.router)
app.include_router(profile.router)
app.include_router(catalog.router)
app.include_router(admin.router)


# ====================================================================
# §7  Static UI — last, because the SPA shell is the broadest match here
#     and `GET /` must not shadow anything above it.
# ====================================================================

static_dir = Path(__file__).parent / "static"


class _RevalidatingStaticFiles(StaticFiles):
    """StaticFiles that makes the browser check before reusing a cached asset.

    Starlette sends ETag and Last-Modified but no Cache-Control, which leaves
    Chrome free to apply *heuristic* freshness: it reuses a cached response for
    a fraction of its age without asking the server at all. For files that are
    edited in place under stable names — app.js, app.css, render.js, none of
    which carry a content hash — that means a tab can keep rendering a UI from
    weeks ago against a server that has moved on, and a plain reload will not
    dislodge it.

    `no-cache` does not disable caching; it requires revalidation. The browser
    still sends If-None-Match and still gets a 304 with an empty body when
    nothing changed, so the cost is one conditional request per asset and the
    UI can never silently be a version behind the server.
    """

    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        resp.headers.setdefault("Cache-Control", "no-cache")
        return resp


if static_dir.exists():
    app.mount("/static", _RevalidatingStaticFiles(directory=str(static_dir)), name="static")


# The one line in index.html that carries the deployment's sub-path. Matched
# literally rather than by regex so a mismatch is a loud "prefix never got
# injected" in the browser, not a silently half-applied rewrite.
_BASE_PATH_MARKER = 'window.ST_BASE_PATH = "";'


def _shell_html(index_file: Path) -> str:
    """index.html with `path_prefix` baked into its ST_BASE_PATH line.

    The UI is static files with no build step, so there is nowhere else to put
    a deployment-time value: the alternative is the browser guessing its own
    prefix from `location.pathname`, which index.html still does as a fallback
    but which is only ever a guess. One `str.replace` on a few KB, per hit on
    `/`, is not worth caching — and not caching is what keeps editing the file
    during dev behave the way the rest of the static tree does.

    An empty prefix rewrites the line to itself; the marker is still asserted
    so a rename in index.html surfaces in the log rather than at the next
    deploy under a virtual path.
    """
    html = index_file.read_text(encoding="utf-8")
    if _BASE_PATH_MARKER not in html:
        logger.warning(
            "shell_base_path_marker_missing",
            extra={"marker": _BASE_PATH_MARKER, "path_prefix": _settings.path_prefix},
        )
        return html
    # json.dumps, not an f-string: this lands inside a <script>, and a prefix
    # with a quote in it would otherwise end the string and run as code.
    return html.replace(
        _BASE_PATH_MARKER,
        f"window.ST_BASE_PATH = {json.dumps(_settings.path_prefix)};",
    )


@app.get("/")
async def index():
    """The UI shell. Same no-cache reasoning as the static mount above — this
    one matters most, since a stale index.html pins every asset it references."""
    index_file = static_dir / "index.html"
    if index_file.exists():
        return HTMLResponse(
            _shell_html(index_file), headers={"Cache-Control": "no-cache"}
        )
    return {"message": "Survey Tagger API is running. See /docs."}


if __name__ == "__main__":
    import uvicorn

    # The import string must match this file's module name. It is resolved by
    # uvicorn at startup, not by Python at import, so a stale name here fails
    # AFTER the whole app has booted — "Could not import module" arrives on the
    # line below a successful taxonomy/tagger/LLM startup log, which reads like
    # anything but a filename. Renaming this module means editing this string.
    #
    # log_level follows the app's own setting rather than being pinned to
    # "debug" — this path re-imports the module (see log_config.configure_logging
    # on why that matters), and a hardcoded debug level here meant every
    # `python run.py` run wrote a firehose to app.log regardless of .env.
    app_env = os.environ.get("APP_ENV", "").strip().lower()
    is_local = "local" in app_env
    port = int(os.environ.get("API_PORT", "8001"))
    log_level = str(_boot_settings.log_level).lower()

    # Which URL actually serves the UI. With a `path_prefix` configured the shell
    # bakes it into ST_BASE_PATH, so the browser loads its assets from under the
    # prefix — PathPrefixMiddleware makes both spellings work, but only one of
    # them matches what the address bar will show afterwards, and guessing wrong
    # is how "the UI only works on IIS" starts. Printed rather than logged: it is
    # for the person who just typed the command.
    _prefix = _boot_settings.path_prefix
    print(f"  UI    http://127.0.0.1:{port}{_prefix or ''}/")
    print(f"  Docs  http://127.0.0.1:{port}{_prefix or ''}/docs")
    if _prefix:
        print(f"  (SURVEY_TAGGER_PATH_PREFIX={_prefix} — the origin root serves the "
              "same app and redirects itself under the prefix.)")

    if not is_local:
        host = os.environ.get("SERVER_HOST", "127.0.0.1")
        uvicorn.run(
            "run:app",
            host=host,
            port=port,
            reload=False,
            log_level=log_level,
            proxy_headers=True,
            forwarded_allow_ips="*",
        )
    else:
        uvicorn.run(
            "run:app",
            host="0.0.0.0",
            port=port,
            reload=False,
            log_level=log_level,
        )
