"""Standalone FastAPI service for the Survey Auto-Tagger.

The single entry point. All wiring comes from `bootstrap.build_context()`; all
tagging work goes through `service.py` (which drives the per-survey engine for a
single survey and the orchestrator for tenant-level work).

Surface (all under /api). The sections below are the order the routes appear in
this file; see "Route layout" further down for why that order is what it is.

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

import asyncio
import errno
import json
import logging
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

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import auth
import discovery
import request_context
import service
import sharefs
from bootstrap import build_context
from projections.survey_view import build_survey_view

import usage_log
from log_config import attach_uvicorn_handlers, configure_logging
from settings import Settings

_boot_settings = Settings()
configure_logging(_boot_settings.log_level, settings=_boot_settings)
logger = logging.getLogger("survey_tagging.api")

# Single composition root for the whole process.
_ctx = build_context()
_settings = _ctx.settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Re-assert the file handler on uvicorn's loggers. When the server is
    # started as `python run.py`, uvicorn installs its own logging config
    # *after* this module was imported and would otherwise leave app.log with
    # no access lines in it.
    attach_uvicorn_handlers()

    # Load the embedding model now, off the request path.
    #
    # `import sentence_transformers` costs ~98s on a cold process (plus ~6s to
    # construct the model and a first forward pass). Left lazy, that whole ~105s
    # lands inside whichever request tags first — which is how a single survey
    # with 7 questions came to take 143 seconds, 66% of it a Python import. It
    # is charged once per process either way; this just moves it somewhere it
    # does not block a caller.
    #
    # On a thread, and never awaited: uvicorn will not serve a request until
    # lifespan startup returns, so blocking here would trade a slow first tag
    # for a two-minute-dead server (including /api/health/share and the UI).
    # Anything that needs the model still calls _load(), which blocks on the
    # same lock until this finishes — so a request arriving mid-warmup waits
    # exactly as long as it would have anyway, and never longer.
    if not _settings.skip_llm:
        import threading

        from llm.embeddings import EmbeddingModel

        threading.Thread(
            target=EmbeddingModel.warm,
            args=(_settings.embedding_model,),
            name="embedding-warmup",
            daemon=True,
        ).start()

    # Start the env-gated periodic auto-retag scheduler (OFF by default).
    from scheduler import AutoRetagScheduler
    app.state.scheduler = AutoRetagScheduler(_ctx)
    await app.state.scheduler.start()
    try:
        yield
    finally:
        await app.state.scheduler.stop()


app = FastAPI(title="Survey Auto-Tagger", version="7.0", lifespan=lifespan)

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

    The UI reads the platform's `access_token` out of localStorage and sends it
    on every call; this is what lets anything downstream (apismx today) forward
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
# §1  Survey — one survey (tenant + survey number)
#     POST tag -> POST retag -> GET tags -> DELETE tags, then the tenant-less
#     ad-hoc upload route.
# ====================================================================

def _survey_view_etag(path: Path, include_journey_candidates: bool) -> str:
    """Strong ETag from file identity (mtime_ns + size); cheap, no body hash."""
    st = sharefs.stat(path)
    base = f'"{st.st_mtime_ns:x}-{st.st_size:x}"'
    return base if not include_journey_candidates else base[:-1] + '+jc"'


@app.post("/api/tenants/{tenant_id}/surveys/{survey_no}/tag")
@app.post("/api/surveys/{survey_no}/tag")
async def tag_survey(
    survey_no: int,
    tenant_id: int | None = None,
    authorization: str | None = Header(default=None),
) -> dict:
    """Tag one survey (incremental — skips if inputs unchanged)."""
    tenant_id = auth.resolve_tenant_id(tenant_id, authorization)
    try:
        return await asyncio.to_thread(service.tag_survey, _ctx, tenant_id, survey_no, force=False)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:  # noqa: BLE001
        logger.exception("tag_survey_failed")
        raise HTTPException(500, f"Tag failed: {e}")


@app.post("/api/tenants/{tenant_id}/surveys/{survey_no}/retag")
@app.post("/api/surveys/{survey_no}/retag")
async def retag_survey(
    survey_no: int,
    tenant_id: int | None = None,
    authorization: str | None = Header(default=None),
) -> dict:
    """Force re-tag one survey (ignore change detection)."""
    tenant_id = auth.resolve_tenant_id(tenant_id, authorization)
    try:
        return await asyncio.to_thread(service.tag_survey, _ctx, tenant_id, survey_no, force=True)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:  # noqa: BLE001
        logger.exception("retag_survey_failed")
        raise HTTPException(500, f"Retag failed: {e}")


@app.get("/api/tenants/{tenant_id}/surveys/{survey_no}/tags")
@app.get("/api/surveys/{survey_no}/tags")
async def get_survey_tags(
    survey_no: int,
    tenant_id: int | None = None,
    include_journey_candidates: bool = False,
    if_none_match: str | None = Header(default=None, alias="If-None-Match"),
    authorization: str | None = Header(default=None),
) -> Response:
    """Unified per-survey view: project tags + question tags + journey rollup.

    `include_journey_candidates=true` surfaces the per-question coverage_metadata
    (ranked canon candidates with scores). ETag/`If-None-Match` → 304.
    """
    tenant_id = auth.resolve_tenant_id(tenant_id, authorization)
    path = service.tagged_output_path(_ctx, tenant_id, survey_no)
    if not sharefs.exists(path):
        raise HTTPException(
            404,
            f"No tagged output for tenant={tenant_id} survey={survey_no}. "
            f"POST /api/tenants/{tenant_id}/surveys/{survey_no}/retag to build it.",
        )

    etag = _survey_view_etag(path, include_journey_candidates)
    if if_none_match and if_none_match.strip() == etag:
        return Response(status_code=304, headers={"ETag": etag})

    try:
        tagged = json.loads(sharefs.read_text(path, encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise HTTPException(500, f"Failed to read tagged output: {e}") from e

    payload = build_survey_view(tagged, include_journey_candidates=include_journey_candidates)
    return JSONResponse(
        content=payload,
        headers={"ETag": etag, "Cache-Control": "private, must-revalidate"},
    )


@app.delete("/api/tenants/{tenant_id}/surveys/{survey_no}/tags")
@app.delete("/api/surveys/{survey_no}/tags")
async def delete_survey_tags(
    survey_no: int,
    tenant_id: int | None = None,
    authorization: str | None = Header(default=None),
) -> dict:
    """Delete tagged output for one survey."""
    tenant_id = auth.resolve_tenant_id(tenant_id, authorization)
    result = service.delete_tagged(_ctx, tenant_id, survey_no)
    if not result["tagged_removed"]:
        raise HTTPException(404, f"No tagged output for tenant={tenant_id} survey={survey_no}")
    return result


@app.post("/api/tag")
async def tag_uploaded(
    survey_file: UploadFile | None = File(None),
    survey_text: str = Form(""),
    industry: str = Form(""),
    company_name: str = Form(""),
    department: str = Form(""),
    purpose: str = Form(""),
    country: str = Form(""),
) -> dict:
    """Ad-hoc: tag a survey from an uploaded JSON file or pasted text (no tenant
    on disk, deterministic, not persisted)."""
    survey_json = None
    if survey_file and survey_file.filename:
        content = await survey_file.read()
        try:
            survey_json = json.loads(content.decode("utf-8-sig"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise HTTPException(400, f"Invalid JSON in uploaded file: {e}")
    elif survey_text.strip():
        try:
            survey_json = json.loads(survey_text)
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"Invalid JSON in pasted text: {e}")
    else:
        raise HTTPException(400, "Provide either a JSON file upload or pasted JSON text")

    overrides = {}
    for k, v in (("industry", industry), ("company_name", company_name),
                 ("department", department), ("purpose", purpose), ("country", country)):
        if v:
            overrides[k] = v
    try:
        return await asyncio.to_thread(service.tag_uploaded, _ctx, survey_json, overrides or None)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, f"Failed to parse survey structure: {e}")


# ====================================================================
# §2  Tenant surveys — §1's work fanned out over every survey
#     The two writers first (incremental, then forced), then the two readers.
#     `/tag-surveys/stream` trails `/tag-surveys` because it is the same
#     listing in a second representation, not a second listing.
# ====================================================================

@app.post("/api/tenants/{tenant_id}/tag-surveys")
@app.post("/api/tag-surveys")
async def tag_tenant_surveys(
    tenant_id: int | None = None,
    authorization: str | None = Header(default=None),
) -> dict:
    """Tag every survey under a tenant (bounded-parallel; incremental)."""
    tenant_id = auth.resolve_tenant_id(tenant_id, authorization)
    return await asyncio.to_thread(service.tag_tenant_surveys, _ctx, tenant_id, force=False)


@app.post("/api/tenants/{tenant_id}/retag-surveys")
@app.post("/api/retag-surveys")
async def retag_tenant_surveys(
    tenant_id: int | None = None,
    authorization: str | None = Header(default=None),
) -> dict:
    """Force re-tag every survey under a tenant."""
    tenant_id = auth.resolve_tenant_id(tenant_id, authorization)
    return await asyncio.to_thread(service.tag_tenant_surveys, _ctx, tenant_id, force=True)


@app.get("/api/tenants/{tenant_id}/tag-surveys")
@app.get("/api/tag-surveys")
async def tenant_tag_status(
    tenant_id: int | None = None,
    authorization: str | None = Header(default=None),
) -> dict:
    """List the tenant's surveys and whether each has tagged output on disk.

    One shot: nothing comes back until every survey has been probed. That is
    fine for a few hundred surveys and is what non-browser callers want; the UI
    uses the `/stream` form below, which does not make the client wait on the
    slowest probe.
    """
    tenant_id = auth.resolve_tenant_id(tenant_id, authorization)
    try:
        surveys = await asyncio.to_thread(service.list_survey_status, _ctx, tenant_id)
    except OSError:
        surveys = []          # unreadable tenant dir reads as "nothing there", as before
    if not surveys:
        raise HTTPException(404, f"No surveys on disk for tenant={tenant_id}")
    return {"tenant_id": tenant_id, "surveys": surveys}


@app.get("/api/tenants/{tenant_id}/tag-surveys/stream")
@app.get("/api/tag-surveys/stream")
async def tenant_tag_status_stream(
    tenant_id: int | None = None,
    authorization: str | None = Header(default=None),
) -> StreamingResponse:
    """The same listing as NDJSON, one survey per line as its probe lands.

    Why this exists: a tenant with thousands of surveys costs one share round
    trip per survey, and the one-shot form above answers only after the last of
    them — long enough that the browser, or a reverse proxy in front of it,
    times out and the listing can never be loaded at all. Here the first line
    leaves as soon as the directory listing is in and the rest trickle out, so
    the connection is never idle and the UI can paint rows as they arrive.

    Lines (`application/x-ndjson`):
        {"kind":"start","tenant_id":75885,"scanning":1240}
        {"kind":"survey","survey_no":12,"tagged":true}      ... completion order
        {"kind":"ping"}                                     ... keepalive only
        {"kind":"done","count":1187}

    The directory listing happens **before** the response starts so a dead share
    is still an HTTP 503 — once the first byte is out, the status code is spent
    and the only way left to report a failure is a line in the body. An empty
    tenant is a normal 200 with `done.count = 0`, not a 404: that keeps 404
    meaning "this server has no such route", which is what lets an older client
    fall back to the one-shot form.
    """
    tenant_id = auth.resolve_tenant_id(tenant_id, authorization)
    try:
        dirs = await asyncio.to_thread(
            discovery.list_survey_dirs, _settings.data_dir, tenant_id
        )
    except OSError as e:
        # A tenant dir that simply is not there is an ordinary empty listing —
        # the same answer the caller gets for a tenant with no surveys. Anything
        # else (rejected logon, unreachable server, timeout) is the share being
        # broken and must not be dressed up as "this corp has nothing".
        # `errno`, not the exception class: smbprotocol raises its own OSError
        # subclass, which bypasses Python's errno->FileNotFoundError mapping.
        if e.errno in (errno.ENOENT, errno.ENOTDIR):
            dirs = []
        else:
            raise HTTPException(503, f"Cannot read the data share: {e}")
    except Exception as e:  # noqa: BLE001 — transport-level refusals are not OSError
        raise HTTPException(503, f"Cannot read the data share: {type(e).__name__}: {e}")

    async def body():
        try:
            async for event in service.stream_survey_status(_ctx, tenant_id, dirs):
                yield json.dumps(event, separators=(",", ":")).encode() + b"\n"
        except Exception as e:  # noqa: BLE001 — headers are gone; the body is the channel
            logger.exception("survey_stream_failed", extra={"tenant_id": tenant_id})
            yield json.dumps({"kind": "error",
                              "detail": f"{type(e).__name__}: {e}"}).encode() + b"\n"

    return StreamingResponse(
        body(),
        media_type="application/x-ndjson",
        # A buffering proxy would hold the whole body and undo the point of
        # this route; nginx/IIS honour X-Accel-Buffering.
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


# ====================================================================
# §3  Tenant tags — tenant_tags.json
#     Plain build -> read -> delete. These read the §4 profile, so they sit
#     above it in dependency order but below the survey routes, which are what
#     callers reach for first.
# ====================================================================

@app.post("/api/tenants/{tenant_id}/tag")
@app.post("/api/tags")
async def tag_tenant(
    tenant_id: int | None = None,
    authorization: str | None = Header(default=None),
) -> dict:
    """Build + persist tenant-level tags (tenant taggers + Parallel.ai profile).

    The tenant-less form is `POST /api/tags`, not `/api/tag` — that one is
    already the ad-hoc "tag this uploaded survey JSON" endpoint.
    """
    tenant_id = auth.resolve_tenant_id(tenant_id, authorization)
    result = await asyncio.to_thread(service.tag_tenant_tags, _ctx, tenant_id)
    if result.get("tenant_tags") is None:
        raise HTTPException(
            422,
            f"No tenant tags produced for tenant={tenant_id} "
            f"(no Parallel.ai profile fetched yet?).",
        )
    return result


@app.get("/api/tenants/{tenant_id}/tags")
@app.get("/api/tags")
async def get_tenant_tags(
    tenant_id: int | None = None,
    authorization: str | None = Header(default=None),
) -> dict:
    """Read tenant_tags.json."""
    tenant_id = auth.resolve_tenant_id(tenant_id, authorization)
    artifact = service.read_tenant_tags(_ctx, tenant_id)
    if artifact is None:
        raise HTTPException(
            404,
            f"No tenant tags for tenant={tenant_id}. "
            f"POST /api/tenants/{tenant_id}/tag to build them.",
        )
    return artifact


@app.delete("/api/tenants/{tenant_id}/tags")
@app.delete("/api/tags")
async def delete_tenant_tags(
    tenant_id: int | None = None,
    authorization: str | None = Header(default=None),
) -> dict:
    """Delete tenant_tags.json."""
    tenant_id = auth.resolve_tenant_id(tenant_id, authorization)
    result = service.delete_tenant_tags(_ctx, tenant_id)
    if not result["removed"]:
        raise HTTPException(404, f"No tenant tags for tenant={tenant_id}")
    return result


# ====================================================================
# §4  Tenant profile — org/cx/ex artifacts under {data_dir}/{t}/tenant_profile/
#     (an INPUT to tagging; `settings.profile_root`, not output_dir)
#     The producer is chosen by `profile_source` (Parallel.ai research vs
#     apismx), but the routes, paths and envelope shape are identical either
#     way, so there is one section rather than two.
#
#     Order: fetch (write) -> profile (read summary) -> diagnose -> {agent}
#     -> delete. `diagnose` MUST precede `{agent}` (rule 1); everything else
#     is lifecycle order.
# ====================================================================

_PARALLEL_AGENTS = ("org", "cx", "ex")


class TenantProfileFetchRequest(BaseModel):
    website: str = Field(
        "", description="Tenant website URL (e.g. https://acme.com). Required when "
                        "profile_source='parallel'; ignored for 'smx', which reads a "
                        "profile the Research API already generated.",
    )
    agents: list[str] | None = Field(
        None, description="Subset of ['org','cx','ex']. None or empty = all three.",
    )
    force: bool = False
    allow_generate: bool = Field(
        True, description="smx only: when the share and apismx both come up empty, "
                          "trigger /AIAccountProfile/Generate and wait for it. "
                          "Set false to look only, never start research.",
    )


def _build_parallel_client():
    from tenant_profile.parallel_client import ParallelClient, ParallelClientError
    if not _settings.parallel_api_key:
        raise HTTPException(
            400, "PARALLEL_API_KEY not set. Add SURVEY_TAGGER_PARALLEL_API_KEY=... to .env.",
        )
    try:
        return ParallelClient(
            api_key=_settings.parallel_api_key,
            processor=_settings.parallel_processor,
            api_timeout=_settings.parallel_api_timeout,
            max_retries=_settings.parallel_max_retries,
        )
    except ParallelClientError as e:
        raise HTTPException(500, f"Parallel.ai client init failed: {e}") from e


def _normalize_agents(agents: list[str] | None) -> tuple[str, ...]:
    if not agents:
        return _PARALLEL_AGENTS
    seen: set[str] = set()
    out: list[str] = []
    for a in agents:
        norm = str(a).strip().lower()
        if norm not in _PARALLEL_AGENTS:
            raise HTTPException(400, f"Unknown agent: {a!r}. Allowed: {list(_PARALLEL_AGENTS)}")
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
    return tuple(out)


def _summarize_batch(result) -> dict:
    return {
        "fetched": [{"tenant_id": r.tenant_id, "agent": r.agent,
                     "artifact_path": str(r.artifact_path)} for r in result.successes],
        "cache_hits": [{"tenant_id": r.tenant_id, "agent": r.agent,
                        "artifact_path": str(r.artifact_path)} for r in result.cache_hits],
        "failures": [{"tenant_id": t, "agent": a, "error": e} for (t, a, e) in result.failures],
        "counts": {"fetched": len(result.successes),
                   "cache_hits": len(result.cache_hits),
                   "failures": len(result.failures)},
    }


def _run_parallel_fetch(tenant_id: int, website: str, agents: tuple[str, ...], force: bool) -> dict:
    from tenant_profile.batch import TenantSpec, DEFAULT_AGENTS, run_batch
    client = _build_parallel_client()
    spec = TenantSpec(tenant_id=tenant_id, website_url=website, agents=DEFAULT_AGENTS)
    result = run_batch(
        specs=[spec], output_dir=_settings.profile_root, client=client, force=force,
        only=agents if set(agents) != set(_PARALLEL_AGENTS) else None, skip=(),
    )
    return _summarize_batch(result)


def _run_smx_fetch(tenant_id: int, website: str, agents: tuple[str, ...],
                   force: bool, token: str, allow_generate: bool) -> dict:
    """Resolve the tenant's profile: share -> apismx /Details -> /Generate.

    Same artifacts, same paths as the Parallel path — only the producer differs.
    """
    from tenant_profile.smx_client import SmxClientError
    from tenant_profile.smx_runner import build_client, resolve_tenant_profile

    try:
        client = build_client(_settings, token)
    except SmxClientError as e:
        raise HTTPException(400, str(e)) from e

    try:
        with client:
            outcome = resolve_tenant_profile(
                tenant_id, _settings.profile_root, client,
                website_url=website, agents=agents, force=force,
                allow_generate=allow_generate and _settings.smx_allow_generate,
                poll_attempts=_settings.smx_generate_poll_attempts,
                poll_interval=_settings.smx_generate_poll_interval,
            )
    except SmxClientError as e:
        raise HTTPException(502, f"apismx fetch failed: {e}") from e

    if outcome.source == "generating":
        # Not a failure: research is running server-side. 202 tells the UI to
        # poll rather than to show an error.
        return {
            "source": "smx", "resolved_via": outcome.source, "pending": True,
            "detail": outcome.detail, "generate": outcome.generate_summary,
            "fetched": [], "cache_hits": [], "failures": [],
            "counts": {"fetched": 0, "cache_hits": 0, "failures": 0},
        }
    if not outcome.ok:
        raise HTTPException(404, outcome.detail or (
            f"apismx has no profile for tenant {tenant_id} and generation was "
            f"not attempted."))

    fetched = [r for r in outcome.results if not r.cached]
    cached = [r for r in outcome.results if r.cached]
    return {
        "source": "smx",
        "resolved_via": outcome.source,
        "pending": False,
        "generate": outcome.generate_summary,
        "fetched": [{"tenant_id": r.tenant_id, "agent": r.agent,
                     "artifact_path": str(r.artifact_path)} for r in fetched],
        "cache_hits": [{"tenant_id": r.tenant_id, "agent": r.agent,
                        "artifact_path": str(r.artifact_path)} for r in cached],
        "failures": [],
        "counts": {"fetched": len(fetched), "cache_hits": len(cached), "failures": 0},
    }


def _run_profile_fetch(tenant_id: int, website: str, agents: tuple[str, ...],
                       force: bool, token: str = "",
                       allow_generate: bool = True) -> dict:
    """Dispatch to whichever producer `profile_source` selects."""
    if _settings.profile_source == "smx":
        return _run_smx_fetch(tenant_id, website, agents, force, token, allow_generate)
    if not website or len(website) < 4:
        raise HTTPException(
            400, "profile_source='parallel' needs a `website` to research.",
        )
    return _run_parallel_fetch(tenant_id, website, agents, force)


def _preflight_profile_fetch(website: str, token: str) -> None:
    """Reject a fetch that cannot start, before the route commits to running it.

    `?background=true` answers 202 and then runs fire-and-forget, so every
    failure after that point is visible only in app.log while the UI polls for
    artifacts that will never appear — for up to 40 minutes. A missing bearer
    token is exactly that shape and is knowable now, so it is raised here as the
    same 400 the synchronous path gives. Cheap and side-effect free: it resolves
    the token and checks the website, and opens no connection.
    """
    if _settings.profile_source == "smx":
        from tenant_profile.smx_client import SmxClientError
        from tenant_profile.smx_runner import resolve_smx_token

        try:
            resolve_smx_token(_settings, token)
        except SmxClientError as e:
            raise HTTPException(400, str(e)) from e
        return
    if not website or len(website) < 4:
        raise HTTPException(400, "profile_source='parallel' needs a `website` to research.")


def _bearer(authorization: str | None) -> str:
    """The caller's raw JWT, for forwarding to apismx. Empty when absent.

    Still passed explicitly on the fetch route rather than read from
    `request_context`: the `?background=true` branch outlives the request, and
    an explicit capture is what makes that safe to read at a glance.
    """
    return auth.bearer_token(authorization)


def _agent_artifact_path(tenant_id: int, agent: str) -> Path:
    from tenant_profile.runner import artifact_path
    return artifact_path(tenant_id, agent, _settings.profile_root)


def _run_smx_diagnose(tenant_id: int, token: str) -> dict:
    """Everything we can learn about one tenant's profile WITHOUT writing anything.

    `/profile/fetch` answers "did it work"; when it did not, this answers why.
    It reads the three sources the cascade consults — the share, `/Details`, and
    `/List` — and reports them side by side. `/List` is the one the cascade never
    looks at and the one that usually holds the answer: `canGenerate`,
    `errorMessage`, and whether the account has a real `WebsiteUrl` or an
    `EffectiveUrl` guessed from its email domain. A `/Generate` that 500s for one
    corp and works for others is nearly always an account with nothing to
    research.

    Strictly read-only: no Generate, no artifacts written. Safe to hit repeatedly.
    """
    from tenant_profile.smx_client import SmxClientError
    from tenant_profile.smx_runner import build_client

    out: dict = {
        "tenant_id": tenant_id,
        "profile_source": _settings.profile_source,
        "on_disk": {a: sharefs.exists(_agent_artifact_path(tenant_id, a))
                    for a in _PARALLEL_AGENTS},
        "apismx": {},
        "account": None,
        "next_step": "",
    }
    if _settings.profile_source != "smx":
        out["next_step"] = (f"profile_source={_settings.profile_source!r} — this "
                            f"diagnostic only covers the smx path.")
        return out

    try:
        client = build_client(_settings, token)
    except SmxClientError as e:
        # No token is the caller's problem, not the upstream's — same 400 the
        # fetch route gives, rather than a 502 blaming apismx.
        raise HTTPException(400, str(e)) from e

    with client:
        try:
            rows = client.get_details(tenant_id)
            out["apismx"]["rows"] = [
                {"profile_type": r.profile_type, "profile_type_name": r.profile_type_name,
                 "agent": r.agent, "is_success": r.is_success,
                 "api_status_message": r.api_status_message,
                 "payload_parsed": r.payload is not None, "created_at": r.created_at}
                for r in rows
            ]
        except SmxClientError as e:
            out["apismx"]["error"] = str(e)
            rows = []

        account = None
        try:
            listed, _meta = client.list_accounts(search=str(tenant_id), page_size=50)
            account = next((r for r in listed if r.corporate_no == tenant_id), None)
            if account is not None:
                out["account"] = {
                    "corporate_no": account.corporate_no,
                    "corporate_id": account.corporate_id,
                    "website_url": account.website_url,
                    "effective_url": account.effective_url,
                    "website_is_derived": account.website_is_derived,
                    "package_name": account.package_name,
                    "account_status": account.account_status,
                    "status": account.status,
                    "can_generate": account.can_generate,
                    "error_message": account.error_message,
                }
        except SmxClientError as e:
            out["list_error"] = str(e)

    out["next_step"] = _diagnose_next_step(tenant_id, out, rows, account)
    return out


def _diagnose_next_step(tenant_id: int, out: dict, rows: list, account) -> str:
    """The one sentence a reader actually wants out of the three source dumps."""
    if all(out["on_disk"].values()):
        return "All three artifacts are already on the share. Nothing to do."
    if any(r.is_success and r.payload is not None for r in rows):
        return (f"apismx has a usable profile — POST /api/tenants/{tenant_id}"
                f"/profile/fetch to persist it to the share.")
    if rows:
        return ("apismx has rows but none are usable (isSuccess false or an "
                "unreadable profileResponse). Regeneration is needed; see "
                "`apismx.rows[].api_status_message`.")
    if account is None:
        return ("No profile in apismx, and /AIAccountProfile/List did not return "
                f"corp {tenant_id} for that search. Confirm the corp exists in "
                "this environment (SOGO_HOST) and that the token can see it.")
    if not account.can_generate:
        return (f"No profile, and the account reports canGenerate=false"
                f"{': ' + account.error_message if account.error_message else ''}. "
                "Generation must be fixed on the SoGo side.")
    if not account.website_url.strip():
        return ("No profile, and the account has no WebsiteUrl — the service "
                f"would research {account.effective_url or 'nothing'}, guessed "
                "from the email domain. Set a real website on the account before "
                "generating.")
    return ("No profile, but the account looks generatable. If /Generate is "
            "still returning 500, the failure is inside the Research API — quote "
            "the smx_http_error line from app.log to that team.")


@app.post("/api/tenants/{tenant_id}/profile/fetch")
@app.post("/api/profile/fetch")
async def tenant_profile_fetch(
    req: TenantProfileFetchRequest,
    tenant_id: int | None = None, background: bool = False,
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    """Fetch the tenant profile (org/cx/ex) from whichever producer
    `profile_source` selects. Sync by default; `?background=true` returns 202
    and runs fire-and-forget.

    Under profile_source='smx' the caller's Bearer token is forwarded to apismx
    (shared issuer), falling back to SURVEY_TAGGER_SMX_TOKEN.
    """
    tenant_id = auth.resolve_tenant_id(tenant_id, authorization)
    agents = _normalize_agents(req.agents)
    token = _bearer(authorization)
    _preflight_profile_fetch(req.website, token)

    if background:
        async def _run():
            try:
                await asyncio.to_thread(_run_profile_fetch, tenant_id, req.website,
                                        agents, req.force, token, req.allow_generate)
                logger.info("tenant_profile_background_fetch_done", extra={"tenant_id_": tenant_id})
            except Exception as e:  # noqa: BLE001
                logger.exception("tenant_profile_background_fetch_failed",
                                 extra={"tenant_id_": tenant_id, "error": str(e)})
        asyncio.create_task(_run())
        return JSONResponse(status_code=202, content={
            "status": "accepted", "tenant_id": tenant_id, "agents": list(agents),
            "source": _settings.profile_source,
            "force": req.force, "poll_url": f"/api/tenants/{tenant_id}/profile",
            "note": "Fetch running in background. Server restart will lose the job.",
        })

    try:
        summary = await asyncio.to_thread(_run_profile_fetch, tenant_id, req.website,
                                          agents, req.force, token, req.allow_generate)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("tenant_profile_fetch_failed")
        raise HTTPException(500, f"Tenant profile fetch failed: {e}") from e
    return JSONResponse(content={"status": "ok", "tenant_id": tenant_id, **summary})


@app.get("/api/tenants/{tenant_id}/profile")
@app.get("/api/profile")
async def get_tenant_profile(
    tenant_id: int | None = None,
    authorization: str | None = Header(default=None),
) -> dict:
    """Summary of on-disk Parallel.ai artifacts for a tenant."""
    tenant_id = auth.resolve_tenant_id(tenant_id, authorization)
    from models.tenant_profile import TenantProfile
    profile = TenantProfile.load(tenant_id, _settings.profile_root)
    if profile is None or profile.is_empty:
        raise HTTPException(
            404,
            f"No tenant profile for tenant_id={tenant_id}. "
            f"POST /api/tenants/{tenant_id}/profile/fetch to build it.",
        )

    summary = {}
    for attr in (
        "industry_vertical", "industry_sub_vertical", "regulatory_intensity",
        "data_sensitivity", "primary_customer_segment", "relationship_type",
        "cx_confidence", "workforce_composition", "work_arrangement",
        "frontline_ratio", "ex_confidence", "corporate_purpose",
    ):
        val = getattr(profile, attr, None)
        if val:
            summary[attr] = val
    if profile.regulatory_frameworks:
        summary["regulatory_frameworks"] = profile.regulatory_frameworks[:10]
    if profile.secondary_customer_segments:
        summary["secondary_customer_segments"] = profile.secondary_customer_segments[:5]
    if profile.customer_types:
        summary["customer_types"] = [
            {"type_name": str(t.get("type_name") or "?")} for t in profile.customer_types[:8]
        ]
    if profile.employee_types:
        summary["employee_types"] = [
            {"type_name": str(t.get("type_name") or "?")} for t in profile.employee_types[:8]
        ]

    return {
        "tenant_id": tenant_id,
        "has_org": profile.has_org, "has_cx": profile.has_cx, "has_ex": profile.has_ex,
        "artifact_paths": profile.artifact_paths,
        "summary": summary,
    }


@app.get("/api/tenants/{tenant_id}/profile/diagnose")
@app.get("/api/profile/diagnose")
async def diagnose_tenant_profile(
    tenant_id: int | None = None,
    authorization: str | None = Header(default=None),
) -> dict:
    """Read-only: share + apismx /Details + apismx /List for one tenant.

    Registered BEFORE `/profile/{agent}` on purpose — FastAPI matches routes in
    definition order, so the other way round `diagnose` would be read as an agent
    name and 400.
    """
    tenant_id = auth.resolve_tenant_id(tenant_id, authorization)
    from tenant_profile.smx_client import SmxClientError
    try:
        return await asyncio.to_thread(_run_smx_diagnose, tenant_id, _bearer(authorization))
    except SmxClientError as e:
        raise HTTPException(502, f"apismx diagnose failed: {e}") from e


@app.get("/api/tenants/{tenant_id}/profile/{agent}")
@app.get("/api/profile/{agent}")
async def get_tenant_profile_agent(
    agent: str,
    tenant_id: int | None = None,
    authorization: str | None = Header(default=None),
) -> dict:
    """Raw envelope JSON for a single Parallel.ai agent (org/cx/ex)."""
    tenant_id = auth.resolve_tenant_id(tenant_id, authorization)
    norm = agent.strip().lower()
    if norm not in _PARALLEL_AGENTS:
        raise HTTPException(400, f"Unknown agent: {agent!r}. Allowed: {list(_PARALLEL_AGENTS)}")
    path = _agent_artifact_path(tenant_id, norm)
    if not sharefs.exists(path):
        raise HTTPException(
            404,
            f"No {norm} artifact for tenant_id={tenant_id}. "
            f"POST /api/tenants/{tenant_id}/profile/fetch with agents=[{norm!r}] to build it.",
        )
    try:
        return json.loads(sharefs.read_text(path, encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise HTTPException(500, f"Failed to read {norm} artifact: {e}") from e


@app.delete("/api/tenants/{tenant_id}/profile")
@app.delete("/api/profile")
async def delete_tenant_profile(
    tenant_id: int | None = None,
    authorization: str | None = Header(default=None),
) -> dict:
    """Delete all Parallel.ai artifacts for a tenant."""
    tenant_id = auth.resolve_tenant_id(tenant_id, authorization)
    profile_dir = _settings.profile_root / str(tenant_id) / "tenant_profile"
    removed: list[str] = []
    if not sharefs.exists(profile_dir):
        return {"tenant_id": tenant_id, "removed": []}
    for agent in _PARALLEL_AGENTS:
        path = _agent_artifact_path(tenant_id, agent)
        if sharefs.exists(path):
            sharefs.unlink(path)
            removed.append(str(path.relative_to(_settings.profile_root)))
    return {"tenant_id": tenant_id, "removed": removed}


# ====================================================================
# §5  Catalog — process-wide, read-only, no tenant in the path
#     Ordered by cost, cheapest first: three in-memory reads (taxonomy,
#     config, me), then one that touches the share (health), then the one
#     that walks the entire share (surveys) and is kept for legacy callers.
# ====================================================================

@app.get("/api/taxonomy")
async def get_taxonomy() -> dict:
    """Full taxonomy — client dropdowns, plus the explanation layer the UI's
    Taxonomy tab renders (`explanation` / `derivation` / `strategy` /
    `purpose` / `feeds`).

    `purpose` is the one-line "why does this exist"; `feeds` names which outcome
    consumes it (S1..S6 for the planned experience-platform services, plus the
    live `Pipeline` / `Reporting` / `None` tokens). See config/taxonomy.yaml's
    header for the token legend.

    Covers all three levels; tenant dims are in here too, so a caller reading a
    tenant_tags.json artifact can look its dimensions up in the same catalog.
    """
    dims = {}
    for name, dim in _ctx.taxonomy.all_dimensions.items():
        dims[name] = {
            "level": dim.level,
            "description": dim.description,
            "purpose": dim.purpose,
            "feeds": dim.feeds,
            "explanation": dim.explanation,
            "derivation": dim.derivation,
            "strategy": dim.strategy,
            "allowed_values": dim.allowed_values,
            "multi_label": dim.multi_label,
            "user_defined": dim.user_defined,
            "canonical_values": dim.canonical_values,
            # Usually empty. Present on crosstab_axis_role, where it publishes
            # the filter/segment/search mapping that `control_role` used to
            # state as a dimension of its own (removed in V7.3).
            "derived_controls": dim.derived_controls,
        }
    return dims


@app.get("/api/config")
async def ui_config() -> dict:
    """Server config the UI needs to shape itself.

    `profile_source` decides the whole tenant-profile panel: the Parallel path
    needs a website and blocks for 10-30 minutes, the SMX path needs neither and
    can trigger generation on a miss. The browser has no other way to know which
    is configured.
    """
    return {
        "profile_source": _settings.profile_source,
        "smx_allow_generate": _settings.smx_allow_generate,
        # Whether apismx can be called with NO caller token — i.e. whether the
        # server holds a headless fallback. The browser knows if it has a token
        # of its own; only this tells it whether the absence is fatal, which is
        # what lets the profile panel refuse up front instead of after a round
        # trip. The token itself is never exposed.
        "smx_token_configured": bool((_settings.smx_token or "").strip()),
        "smx_generate_wait_seconds": int(
            _settings.smx_generate_poll_attempts * _settings.smx_generate_poll_interval
        ),
        # So "the trace is on" is checkable without reading the server's .env.
        "smx_debug_wire": _settings.smx_debug_wire,
        "skip_llm": _settings.skip_llm,
    }


@app.get("/api/me")
async def whoami(authorization: str | None = Header(default=None)) -> dict:
    """What the caller's Bearer token says about them.

    `{corp_no, subject, has_token, verified, auth_enabled}`. The UI calls this
    once at boot: when it is embedded in the platform shell there is a token in
    localStorage but no corp number typed into the box, and this is how it
    learns which tenant to open. A typed corp number always overrides it.

    `verified` is False when the signature could not be checked (no/again wrong
    public key) but the claims were read anyway — see auth._decode_verified. It
    is advisory while `auth_enabled` is False.
    """
    return {**auth.principal(authorization), "auth_enabled": _settings.auth_enabled}


@app.get("/api/health/share")
async def share_health() -> dict:
    """Is the data root reachable? Lets the UI distinguish a downed share from
    a tenant that simply has no surveys (both otherwise look like an empty list)."""
    return await asyncio.to_thread(discovery.probe_root, _settings.data_dir)


@app.get("/api/surveys")
async def list_surveys() -> list[dict]:
    """List all tenants and their surveys discovered under the local data dir.

    Walks the whole data root — expensive over a network share (minutes, not
    seconds). The current UI does not call it; prefer
    GET /api/tenants/{t}/tag-surveys, which is one directory listing.

    Offloaded to a thread even though it is the only caller's own problem how
    long it takes: run inline in the event loop, one request to this route
    freezes the entire server for everyone — no other route answers, not even
    /api/health/share, and the process looks hung rather than busy. A stale
    browser tab still holding the pre-v7 UI calls this on load, which is exactly
    how that happens in practice.
    """
    return await asyncio.to_thread(discovery.discover_catalog, _settings.data_dir)


# ====================================================================
# §6  Admin — the auto-retag scheduler (scheduler.py)
#     Separate from §5 because these are the only non-tenant routes that
#     *do* something: run-now kicks off a full change-scan.
# ====================================================================

@app.get("/api/admin/autoretag")
async def autoretag_status() -> dict:
    """Auto-retag scheduler status (enabled?, interval, last scan)."""
    sched = getattr(app.state, "scheduler", None)
    if sched is None:
        return {"enabled": False, "status": "uninitialized"}
    return sched.status()


@app.post("/api/admin/autoretag/run-now")
async def autoretag_run_now() -> dict:
    """Run one change-scan immediately (works even when the periodic loop is off)."""
    sched = getattr(app.state, "scheduler", None)
    if sched is None:
        raise HTTPException(503, "Scheduler not initialized")
    return await sched.run_once()


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


@app.get("/")
async def index():
    """The UI shell. Same no-cache reasoning as the static mount above — this
    one matters most, since a stale index.html pins every asset it references."""
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file), headers={"Cache-Control": "no-cache"})
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
    uvicorn.run("run:app", host="0.0.0.0", port=8001, reload=False,
                log_level=str(_boot_settings.log_level).lower())
