"""Tenant-profile routes — org/cx/ex artifacts (producer set by `profile_source`).

Lifted out of run.py unchanged — see api/__init__.py for why the split."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import auth
import sharefs
from api import deps

logger = logging.getLogger("survey_tagging.api")

router = APIRouter()


# ====================================================================
# §4  Tenant profile — org/cx/ex artifacts under {data_dir}/{t}/tenant_profile/
#     (an INPUT to tagging; `deps.settings.profile_root`, not output_dir)
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
    """The shared factory, with its error translated into HTTP.

    A missing key is the operator's problem (400 — nothing to research with);
    anything else the SDK raises on construction is ours (500).
    """
    from tenant_profile.parallel_client import ParallelClientError, build_parallel_client

    try:
        return build_parallel_client(deps.settings)
    except ParallelClientError as e:
        status = 400 if "PARALLEL_API_KEY" in str(e) else 500
        raise HTTPException(status, str(e)) from e


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
        specs=[spec], output_dir=deps.settings.profile_root, client=client, force=force,
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
        client = build_client(deps.settings, token)
    except SmxClientError as e:
        raise HTTPException(400, str(e)) from e

    try:
        with client:
            outcome = resolve_tenant_profile(
                tenant_id, deps.settings.profile_root, client,
                website_url=website, agents=agents, force=force,
                allow_generate=allow_generate and deps.settings.smx_allow_generate,
                poll_attempts=deps.settings.smx_generate_poll_attempts,
                poll_interval=deps.settings.smx_generate_poll_interval,
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
    if deps.settings.profile_source == "smx":
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
    if deps.settings.profile_source == "smx":
        from tenant_profile.smx_client import SmxClientError
        from tenant_profile.smx_runner import resolve_smx_token

        try:
            resolve_smx_token(deps.settings, token)
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
    return artifact_path(tenant_id, agent, deps.settings.profile_root)


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
        "profile_source": deps.settings.profile_source,
        "on_disk": {a: sharefs.exists(_agent_artifact_path(tenant_id, a))
                    for a in _PARALLEL_AGENTS},
        "apismx": {},
        "account": None,
        "next_step": "",
    }
    if deps.settings.profile_source != "smx":
        out["next_step"] = (f"profile_source={deps.settings.profile_source!r} — this "
                            f"diagnostic only covers the smx path.")
        return out

    try:
        client = build_client(deps.settings, token)
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


@router.post("/api/tenants/{tenant_id}/profile/fetch")
@router.post("/api/profile/fetch")
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
            "source": deps.settings.profile_source,
            "force": req.force, "poll_url": f"/api/tenants/{tenant_id}/profile",
            "note": "Fetch running in background. Server restart will lose the job.",
        })

    try:
        summary = await asyncio.to_thread(_run_profile_fetch, tenant_id, req.website,
                                          agents, req.force, token, req.allow_generate)
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001
        logger.exception("tenant_profile_fetch_failed")
        raise deps.server_error("Tenant profile fetch failed")
    return JSONResponse(content={"status": "ok", "tenant_id": tenant_id, **summary})


@router.get("/api/tenants/{tenant_id}/profile")
@router.get("/api/profile")
async def get_tenant_profile(
    tenant_id: int | None = None,
    authorization: str | None = Header(default=None),
) -> dict:
    """Summary of on-disk Parallel.ai artifacts for a tenant."""
    tenant_id = auth.resolve_tenant_id(tenant_id, authorization)
    from models.tenant_profile import TenantProfile
    profile = TenantProfile.load(tenant_id, deps.settings.profile_root)
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


@router.get("/api/tenants/{tenant_id}/profile/diagnose")
@router.get("/api/profile/diagnose")
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


@router.get("/api/tenants/{tenant_id}/profile/{agent}")
@router.get("/api/profile/{agent}")
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


@router.delete("/api/tenants/{tenant_id}/profile")
@router.delete("/api/profile")
async def delete_tenant_profile(
    tenant_id: int | None = None,
    authorization: str | None = Header(default=None),
) -> dict:
    """Delete all Parallel.ai artifacts for a tenant."""
    tenant_id = auth.resolve_tenant_id(tenant_id, authorization)
    profile_dir = deps.settings.profile_root / str(tenant_id) / "tenant_profile"
    removed: list[str] = []
    if not sharefs.exists(profile_dir):
        return {"tenant_id": tenant_id, "removed": []}
    for agent in _PARALLEL_AGENTS:
        path = _agent_artifact_path(tenant_id, agent)
        if sharefs.exists(path):
            sharefs.unlink(path)
            removed.append(str(path.relative_to(deps.settings.profile_root)))
    return {"tenant_id": tenant_id, "removed": removed}
