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
    GET    /admin/autoretag        (see scheduler.py)
    POST   /admin/autoretag/run-now
    GET    /                       static UI
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
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

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import discovery
import service
import sharefs
from bootstrap import build_context
from projections.survey_view import build_survey_view

from log_config import configure_logging
from settings import Settings

configure_logging(Settings().log_level)
logger = logging.getLogger("survey_tagging.api")

# Single composition root for the whole process.
_ctx = build_context()
_settings = _ctx.settings


@asynccontextmanager
async def lifespan(app: FastAPI):
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


# ====================================================================
# Survey tagging
# ====================================================================

@app.post("/api/tenants/{tenant_id}/surveys/{survey_no}/tag")
async def tag_survey(tenant_id: int, survey_no: int) -> dict:
    """Tag one survey (incremental — skips if inputs unchanged)."""
    try:
        return await asyncio.to_thread(service.tag_survey, _ctx, tenant_id, survey_no, force=False)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:  # noqa: BLE001
        logger.exception("tag_survey_failed")
        raise HTTPException(500, f"Tag failed: {e}")


@app.post("/api/tenants/{tenant_id}/surveys/{survey_no}/retag")
async def retag_survey(tenant_id: int, survey_no: int) -> dict:
    """Force re-tag one survey (ignore change detection)."""
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
async def get_survey_tags(
    tenant_id: int,
    survey_no: int,
    include_journey_candidates: bool = False,
    if_none_match: str | None = Header(default=None, alias="If-None-Match"),
) -> Response:
    """Unified per-survey view: project tags + question tags + journey rollup.

    `include_journey_candidates=true` surfaces the per-question coverage_metadata
    (ranked canon candidates with scores). ETag/`If-None-Match` → 304.
    """
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
async def delete_survey_tags(tenant_id: int, survey_no: int) -> dict:
    """Delete tagged output for one survey."""
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
async def tag_tenant_surveys(tenant_id: int) -> dict:
    """Tag every survey under a tenant (bounded-parallel; incremental)."""
    return await asyncio.to_thread(service.tag_tenant_surveys, _ctx, tenant_id, force=False)


@app.post("/api/tenants/{tenant_id}/retag-surveys")
async def retag_tenant_surveys(tenant_id: int) -> dict:
    """Force re-tag every survey under a tenant."""
    return await asyncio.to_thread(service.tag_tenant_surveys, _ctx, tenant_id, force=True)


@app.get("/api/tenants/{tenant_id}/tag-surveys")
async def tenant_tag_status(tenant_id: int) -> dict:
    """List the tenant's surveys and whether each has tagged output on disk."""
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
async def tag_tenant(tenant_id: int) -> dict:
    """Build + persist tenant-level tags (tenant taggers + Parallel.ai profile)."""
    result = await asyncio.to_thread(service.tag_tenant_tags, _ctx, tenant_id)
    if result.get("tenant_tags") is None:
        raise HTTPException(
            422,
            f"No tenant tags produced for tenant={tenant_id} "
            f"(no Parallel.ai profile fetched yet?).",
        )
    return result


@app.get("/api/tenants/{tenant_id}/tags")
async def get_tenant_tags(tenant_id: int) -> dict:
    """Read tenant_tags.json."""
    artifact = service.read_tenant_tags(_ctx, tenant_id)
    if artifact is None:
        raise HTTPException(
            404,
            f"No tenant tags for tenant={tenant_id}. "
            f"POST /api/tenants/{tenant_id}/tag to build them.",
        )
    return artifact


@app.delete("/api/tenants/{tenant_id}/tags")
async def delete_tenant_tags(tenant_id: int) -> dict:
    """Delete tenant_tags.json."""
    result = service.delete_tenant_tags(_ctx, tenant_id)
    if not result["removed"]:
        raise HTTPException(404, f"No tenant tags for tenant={tenant_id}")
    return result


# ====================================================================
# Parallel.ai tenant profile
# ====================================================================

_PARALLEL_AGENTS = ("org", "cx", "ex")


class TenantProfileFetchRequest(BaseModel):
    website: str = Field(..., min_length=4,
                         description="Tenant website URL (e.g. https://acme.com)")
    agents: list[str] | None = Field(
        None, description="Subset of ['org','cx','ex']. None or empty = all three.",
    )
    force: bool = False


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


def _agent_artifact_path(tenant_id: int, agent: str) -> Path:
    from tenant_profile.runner import artifact_path
    return artifact_path(tenant_id, agent, Path(_settings.output_dir))


@app.post("/api/tenants/{tenant_id}/profile/fetch")
async def tenant_profile_fetch(
    tenant_id: int, req: TenantProfileFetchRequest, background: bool = False,
) -> JSONResponse:
    """Fetch the Parallel.ai tenant profile (org/cx/ex). Sync by default;
    `?background=true` returns 202 and runs fire-and-forget."""
    agents = _normalize_agents(req.agents)

    if background:
        async def _run():
            try:
                await asyncio.to_thread(_run_parallel_fetch, tenant_id, req.website, agents, req.force)
                logger.info("tenant_profile_background_fetch_done", extra={"tenant_id_": tenant_id})
            except Exception as e:  # noqa: BLE001
                logger.exception("tenant_profile_background_fetch_failed",
                                 extra={"tenant_id_": tenant_id, "error": str(e)})
        asyncio.create_task(_run())
        return JSONResponse(status_code=202, content={
            "status": "accepted", "tenant_id": tenant_id, "agents": list(agents),
            "force": req.force, "poll_url": f"/api/tenants/{tenant_id}/profile",
            "note": "Fetch running in background. Server restart will lose the job.",
        })

    try:
        summary = await asyncio.to_thread(_run_parallel_fetch, tenant_id, req.website, agents, req.force)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("tenant_profile_fetch_failed")
        raise HTTPException(500, f"Tenant profile fetch failed: {e}") from e
    return JSONResponse(content={"status": "ok", "tenant_id": tenant_id, **summary})


@app.get("/api/tenants/{tenant_id}/profile")
async def get_tenant_profile(tenant_id: int) -> dict:
    """Summary of on-disk Parallel.ai artifacts for a tenant."""
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
async def get_tenant_profile_agent(tenant_id: int, agent: str) -> dict:
    """Raw envelope JSON for a single Parallel.ai agent (org/cx/ex)."""
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
async def delete_tenant_profile(tenant_id: int) -> dict:
    """Delete all Parallel.ai artifacts for a tenant."""
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
    """Full taxonomy for populating client dropdowns."""
    dims = {}
    for name, dim in _ctx.taxonomy.all_dimensions.items():
        dims[name] = {
            "level": dim.level,
            "description": dim.description,
            "allowed_values": dim.allowed_values,
            "multi_label": dim.multi_label,
            "user_defined": dim.user_defined,
            "canonical_values": dim.canonical_values,
        }
    return dims


@app.get("/api/surveys")
async def list_surveys() -> list[dict]:
    """List all tenants and their surveys discovered under the local data dir.

    Walks the whole data root — expensive over a network share. The UI does not
    use it; prefer GET /api/tenants/{t}/tag-surveys, which is one dir listing.
    """
    return discovery.discover_catalog(_settings.data_dir)


@app.get("/api/health/share")
async def share_health() -> dict:
    """Is the data root reachable? Lets the UI distinguish a downed share from
    a tenant that simply has no surveys (both otherwise look like an empty list)."""
    return await asyncio.to_thread(discovery.probe_root, _settings.data_dir)


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
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def index():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Survey Tagger API is running. See /docs."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=False, log_level="debug")
