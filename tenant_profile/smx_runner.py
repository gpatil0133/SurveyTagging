"""Persist SMX-sourced tenant profiles as our standard artifacts.

The counterpart to `tenant_profile.runner` for `profile_source="smx"`. It reads
`/AIAccountProfile/Details` once per tenant and writes the same three files to
the same place:

    {output_dir}/{tenant_id}/tenant_profile/{org_profile,cx_intelligence,ex_intelligence}.json

Keeping the on-disk contract identical is what makes this a swap rather than a
migration: `models.TenantProfile`, the tenant taggers, `llm.tenant_canon` and —
critically — `pipeline.change_detector` (which fingerprints this directory to
decide which surveys need re-tagging) all keep working untouched.

The only shape difference between the two producers is `journeys`; see
`adapt_payload`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tenant_profile.runner import (
    ARTIFACT_SCHEMA_VERSION,
    Agent,
    FetchResult,
    artifact_path,
    load_artifact,
    _write_atomic,
)
from tenant_profile.schemas import validate as validate_payload
from tenant_profile.smx_client import ProfileRow, SmxClient, SmxClientError

logger = logging.getLogger(__name__)


def adapt_payload(agent: Agent, payload: dict[str, Any]) -> dict[str, Any]:
    """Reshape an SMX agent payload into the layout our accessors expect.

    org and ex arrive ready to use. CX differs in exactly one place: the agent
    nests the journey list under `journey_analysis.journeys`, while
    `models.TenantProfile.cx_journeys` — and therefore
    `llm.tenant_canon._aggregate_raw_stages` — reads a top-level `journeys`.

    This matters more than its size suggests. Without the hoist nothing raises;
    the canon builder simply aggregates zero stages, the gate falls back to
    `industry_template`, and every `journey_stage` / `sub_stage_name` tag
    silently degrades from tenant-specific to generic. Hence the regression test
    in tests/test_tenant_profile/test_smx_adapter.py.

    The original `journey_analysis` is preserved alongside the hoisted list so
    nothing (complexity, future fields) is lost.
    """
    if agent != "cx":
        return payload
    if payload.get("journeys"):
        return payload
    nested = payload.get("journey_analysis")
    if not isinstance(nested, dict):
        return payload
    journeys = nested.get("journeys")
    if not isinstance(journeys, list) or not journeys:
        return payload
    adapted = dict(payload)
    adapted["journeys"] = journeys
    return adapted


def build_envelope(tenant_id: int, row: ProfileRow, website_url: str,
                   payload: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    """Our standard artifact envelope, with SMX provenance in place of Parallel's."""
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "agent": row.agent,
        "tenant_id": tenant_id,
        "website_url": website_url,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "service": "apismx/AIAccountProfile",
            "profile_type": row.profile_type,
            "profile_type_name": row.profile_type_name,
            "is_success": row.is_success,
            "api_status_message": row.api_status_message,
            "created_at": row.created_at,
        },
        "validation_warnings": warnings,
        "agent_output": payload,
    }


def fetch_tenant_profile(
    tenant_id: int,
    output_dir: Path,
    client: SmxClient,
    *,
    website_url: str = "",
    agents: tuple[str, ...] = ("org", "cx", "ex"),
    force: bool = False,
) -> list[FetchResult]:
    """Read one tenant's profiles from SMX and write them as artifacts.

    Idempotent in the same way as the Parallel runner: an artifact already on
    disk is returned untouched unless `force=True`. One `Details` call covers
    all three agents, so the cache check happens per-agent but the fetch does
    not repeat.

    Raises SmxClientError only for transport/auth failures. A tenant with no
    generated profile is not an error — it returns an empty list, which callers
    surface as "not fetched yet".
    """
    wanted = {a.strip().lower() for a in agents}

    if not force:
        cached = {}
        for agent in wanted:
            envelope = load_artifact(artifact_path(tenant_id, agent, output_dir))  # type: ignore[arg-type]
            if envelope is not None:
                cached[agent] = envelope
        if len(cached) == len(wanted):
            logger.info("smx_artifact_cache_hit_all",
                        extra={"tenant_id": tenant_id, "agents": sorted(wanted)})
            return [
                FetchResult(agent=agent, tenant_id=tenant_id,  # type: ignore[arg-type]
                            artifact_path=artifact_path(tenant_id, agent, output_dir),  # type: ignore[arg-type]
                            cached=True, envelope=envelope)
                for agent, envelope in sorted(cached.items())
            ]

    rows = client.get_details(tenant_id)
    if not rows:
        logger.info("smx_no_profile_generated", extra={"tenant_id": tenant_id})
        return []

    results: list[FetchResult] = []
    for row in rows:
        if row.agent is None or row.agent not in wanted:
            continue
        if not row.is_success or row.payload is None:
            logger.warning(
                "smx_profile_row_unusable",
                extra={"tenant_id": tenant_id, "agent": row.agent,
                       "is_success": row.is_success,
                       "parsed": row.payload is not None,
                       "api_status_message": row.api_status_message},
            )
            continue

        path = artifact_path(tenant_id, row.agent, output_dir)  # type: ignore[arg-type]
        if not force:
            existing = load_artifact(path)
            if existing is not None:
                results.append(FetchResult(agent=row.agent, tenant_id=tenant_id,  # type: ignore[arg-type]
                                           artifact_path=path, cached=True,
                                           envelope=existing))
                continue

        payload = adapt_payload(row.agent, row.payload)  # type: ignore[arg-type]
        warnings = validate_payload(payload, row.agent)
        if warnings:
            logger.warning("smx_artifact_validation_warnings",
                           extra={"tenant_id": tenant_id, "agent": row.agent,
                                  "warnings": warnings})
        envelope = build_envelope(tenant_id, row, website_url, payload, warnings)
        _write_atomic(path, envelope)
        logger.info("smx_artifact_written",
                    extra={"tenant_id": tenant_id, "agent": row.agent,
                           "path": str(path), "warning_count": len(warnings)})
        results.append(FetchResult(agent=row.agent, tenant_id=tenant_id,  # type: ignore[arg-type]
                                   artifact_path=path, cached=False, envelope=envelope))

    missing = wanted - {r.agent for r in results}
    if missing:
        logger.warning("smx_agents_missing",
                       extra={"tenant_id": tenant_id, "agents": sorted(missing)})
    return results


@dataclass
class ResolveResult:
    """Outcome of the disk -> fetch -> generate cascade."""

    results: list[FetchResult]
    # How the profile was obtained:
    #   "disk"        already on the share, nothing called
    #   "smx"         read from /Details
    #   "generated"   /Generate was triggered, then /Details succeeded
    #   "generating"  /Generate accepted but no profile appeared before the deadline
    #   "unavailable" nothing on disk, nothing in SMX, generate not allowed/failed
    source: str
    generate_summary: dict[str, Any] | None = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.results)


def resolve_tenant_profile(
    tenant_id: int,
    output_dir: Path,
    client: SmxClient,
    *,
    website_url: str = "",
    agents: tuple[str, ...] = ("org", "cx", "ex"),
    force: bool = False,
    allow_generate: bool = True,
    poll_attempts: int = 6,
    poll_interval: float = 15.0,
    sleep: Any = time.sleep,
) -> ResolveResult:
    """Resolve a tenant's profile: share first, then SMX, then generate.

        1. Artifacts already on the share    -> use them, no network call
        2. /AIAccountProfile/Details          -> persist and use
        3. /AIAccountProfile/Generate         -> then re-poll Details

    Step 3 is a write that starts paid research, so it only runs when
    `allow_generate` is set and steps 1-2 both came up empty.

    Generation is not instant — three agents took ~44s end to end in the
    observed run — so after triggering we re-poll `Details` rather than
    assuming the rows are there. If the deadline passes without a profile the
    result is "generating", not a failure: the work is still running server-side
    and the next call will pick it up from step 2.
    """
    fetched = fetch_tenant_profile(
        tenant_id, output_dir, client,
        website_url=website_url, agents=agents, force=force,
    )
    if fetched:
        source = "disk" if all(r.cached for r in fetched) else "smx"
        return ResolveResult(results=fetched, source=source)

    if not allow_generate:
        return ResolveResult(
            results=[], source="unavailable",
            detail=f"No profile on the share and none in SMX for tenant {tenant_id}.",
        )

    logger.info("smx_generate_triggered", extra={"tenant_id": tenant_id})
    try:
        summary = client.generate([tenant_id])
    except SmxClientError as e:
        # The route turns this into a 404, whose body the browser sees and the
        # log does not. Say it here too, or the server-side story ends at
        # "smx_generate_triggered" with no outcome. (The full response body is
        # already on the ERROR line SmxClient._request emitted.)
        logger.error("smx_generate_failed",
                     extra={"tenant_id": tenant_id, "error": str(e)})
        return ResolveResult(
            results=[], source="unavailable", detail=f"Generate failed: {e}",
        )

    for attempt in range(1, poll_attempts + 1):
        sleep(poll_interval)
        try:
            fetched = fetch_tenant_profile(
                tenant_id, output_dir, client,
                website_url=website_url, agents=agents, force=force,
            )
        except SmxClientError as e:
            # A transient read failure mid-generation should not abandon the
            # run; the profile may still land on a later attempt.
            logger.warning("smx_poll_failed",
                           extra={"tenant_id": tenant_id, "attempt": attempt,
                                  "error": str(e)})
            continue
        if fetched:
            logger.info("smx_generate_completed",
                        extra={"tenant_id": tenant_id, "attempts": attempt})
            return ResolveResult(results=fetched, source="generated",
                                 generate_summary=summary)

    waited = int(poll_attempts * poll_interval)
    return ResolveResult(
        results=[], source="generating", generate_summary=summary,
        detail=(f"Generation was triggered for tenant {tenant_id} but no profile "
                f"appeared within {waited}s. It is still running server-side — "
                f"retry this fetch shortly."),
    )


def build_client(settings: Any, token: str = "") -> SmxClient:
    """Construct an SmxClient from Settings, preferring a request-scoped token.

    `token` is the inbound caller's JWT when there is one; apismx accepts it
    because it shares an issuer with our own auth. When it is not passed
    explicitly the request-scoped token is used — the browser puts one on every
    call, so a route that never threaded it through still forwards it.
    `settings.smx_token` is the headless fallback (CLI, scheduler).

    This is also the single place the wire trace is configured from Settings, so
    every client built for the app is traced identically and one constructed by
    hand (tests) is not traced at all.
    """
    import request_context
    from tenant_profile.smx_trace import SmxTrace

    verify: bool | str = settings.sogo_verify_ssl
    if settings.sogo_ca_bundle_path:
        verify = settings.sogo_ca_bundle_path
    resolved = (token or request_context.current_token() or settings.smx_token or "").strip()
    if not resolved:
        raise SmxClientError(
            "profile_source='smx' needs a bearer token. Forward the caller's JWT "
            "or set SURVEY_TAGGER_SMX_TOKEN in .env."
        )
    return SmxClient(
        base_url=settings.sogo_apismx_base_url,
        pmx_base_url=settings.sogo_apipmx_base_url,
        token=resolved,
        timeout=settings.smx_request_timeout,
        verify=verify,
        trace=SmxTrace(
            enabled=bool(getattr(settings, "smx_debug_wire", False)),
            log_dir=getattr(settings, "log_dir", None),
            max_chars=int(getattr(settings, "smx_debug_max_chars", 2000) or 2000),
        ),
    )
