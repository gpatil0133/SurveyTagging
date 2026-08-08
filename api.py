"""Standalone FastAPI service for the Survey Auto-Tagger.

The single entry point. All wiring comes from `bootstrap.build_context()`; all
tagging work goes through `service.py` (which drives the per-survey engine for a
single survey and the orchestrator for tenant-level work).

Surface (all under /api):
  Survey
    POST   /tenants/{t}/surveys/{s}/tag        tag one survey (incremental)
    POST   /tenants/{t}/surveys/{s}/retag      force re-tag one survey
    GET    /tenants/{t}/surveys/{s}/tags       unified survey view (ETag/304)
    DELETE /tenants/{t}/surveys/{s}/tags       delete one survey's tags
    POST   /tag                                ad-hoc: tag uploaded survey JSON
  Per-tenant survey tagging
    POST   /tenants/{t}/tag-surveys            tag all surveys (bounded-parallel)
    POST   /tenants/{t}/retag-surveys          force re-tag all surveys
    GET    /tenants/{t}/tag-surveys            tagged-status for the tenant
  Tenant tags
    POST   /tenants/{t}/tag                     build tenant_tags.json
    GET    /tenants/{t}/tags                     read tenant_tags.json
    DELETE /tenants/{t}/tags                     delete tenant_tags.json
  Parallel.ai tenant profile
    POST   /tenants/{t}/profile/fetch
    GET    /tenants/{t}/profile
    GET    /tenants/{t}/profile/{agent}
    DELETE /tenants/{t}/profile
  Catalog / admin
    GET    /taxonomy
    GET    /surveys
    GET    /me                     caller identity from the Bearer token
    GET    /admin/autoretag        (see scheduler.py)
    POST   /admin/autoretag/run-now
    GET    /                       static UI

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
from fastapi.responses import FileResponse, JSONResponse, Response
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
    # started as `python api.py`, uvicorn installs its own logging config
    # *after* this module was imported and would otherwise leave app.log with
    # no access lines in it.
    attach_uvicorn_handlers()

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
# Survey tagging
# ====================================================================

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


def _survey_view_etag(path: Path, include_journey_candidates: bool) -> str:
    """Strong ETag from file identity (mtime_ns + size); cheap, no body hash."""
    st = sharefs.stat(path)
    base = f'"{st.st_mtime_ns:x}-{st.st_size:x}"'
    return base if not include_journey_candidates else base[:-1] + '+jc"'


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
# Per-tenant survey tagging (all surveys)
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
    """List the tenant's surveys and whether each has tagged output on disk."""
    tenant_id = auth.resolve_tenant_id(tenant_id, authorization)
    surveys = []
    for sno in discovery.list_survey_nos(_settings.data_dir, tenant_id):
        tagged = sharefs.exists(service.tagged_output_path(_ctx, tenant_id, sno))
        surveys.append({"survey_no": sno, "tagged": tagged})
    if not surveys:
        raise HTTPException(404, f"No surveys on disk for tenant={tenant_id}")
    return {"tenant_id": tenant_id, "surveys": surveys}


# ====================================================================
# Tenant tags (tenant_tags.json)
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
# Parallel.ai tenant profile
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
        specs=[spec], output_dir=Path(_settings.output_dir), client=client, force=force,
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
                tenant_id, Path(_settings.output_dir), client,
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


def _bearer(authorization: str | None) -> str:
    """The caller's raw JWT, for forwarding to apismx. Empty when absent.

    Still passed explicitly on the fetch route rather than read from
    `request_context`: the `?background=true` branch outlives the request, and
    an explicit capture is what makes that safe to read at a glance.
    """
    return auth.bearer_token(authorization)


def _agent_artifact_path(tenant_id: int, agent: str) -> Path:
    from tenant_profile.runner import artifact_path
    return artifact_path(tenant_id, agent, Path(_settings.output_dir))


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
    profile = TenantProfile.load(tenant_id, Path(_settings.output_dir))
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
    profile_dir = Path(_settings.output_dir) / str(tenant_id) / "tenant_profile"
    removed: list[str] = []
    if not sharefs.exists(profile_dir):
        return {"tenant_id": tenant_id, "removed": []}
    for agent in _PARALLEL_AGENTS:
        path = _agent_artifact_path(tenant_id, agent)
        if sharefs.exists(path):
            sharefs.unlink(path)
            removed.append(str(path.relative_to(Path(_settings.output_dir))))
    return {"tenant_id": tenant_id, "removed": removed}


# ====================================================================
# Catalog / admin
# ====================================================================

@app.get("/api/taxonomy")
async def get_taxonomy() -> dict:
    """Full taxonomy — client dropdowns, plus the explanation layer the UI's
    Taxonomy tab renders (`explanation` / `derivation` / `strategy`).

    Covers all three levels; tenant dims are in here too, so a caller reading a
    tenant_tags.json artifact can look its dimensions up in the same catalog.
    """
    dims = {}
    for name, dim in _ctx.taxonomy.all_dimensions.items():
        dims[name] = {
            "level": dim.level,
            "description": dim.description,
            "explanation": dim.explanation,
            "derivation": dim.derivation,
            "strategy": dim.strategy,
            "allowed_values": dim.allowed_values,
            "multi_label": dim.multi_label,
            "user_defined": dim.user_defined,
            "canonical_values": dim.canonical_values,
        }
    return dims


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


@app.get("/api/health/share")
async def share_health() -> dict:
    """Is the data root reachable? Lets the UI distinguish a downed share from
    a tenant that simply has no surveys (both otherwise look like an empty list)."""
    return await asyncio.to_thread(discovery.probe_root, _settings.data_dir)


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
        "smx_generate_wait_seconds": int(
            _settings.smx_generate_poll_attempts * _settings.smx_generate_poll_interval
        ),
        "skip_llm": _settings.skip_llm,
    }


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


# ---------- Static UI ----------

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
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=False, log_level="debug")
