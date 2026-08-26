"""Tenant-tag routes — tenant_tags.json.

Lifted out of run.py unchanged — see api/__init__.py for why the split."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Header, HTTPException

import auth
import service
from api import deps

logger = logging.getLogger("survey_tagging.api")

router = APIRouter()


# ====================================================================
# §3  Tenant tags — tenant_tags.json
#     Plain build -> read -> delete. These read the §4 profile, so they sit
#     above it in dependency order but below the survey routes, which are what
#     callers reach for first.
# ====================================================================

@router.post("/api/tenants/{tenant_id}/tag")
@router.post("/api/tags")
async def tag_tenant(
    tenant_id: int | None = None,
    authorization: str | None = Header(default=None),
) -> dict:
    """Build + persist tenant-level tags (tenant taggers + Parallel.ai profile).

    The tenant-less form is `POST /api/tags`, not `/api/tag` — that one is
    already the ad-hoc "tag this uploaded survey JSON" endpoint.
    """
    tenant_id = auth.resolve_tenant_id(tenant_id, authorization)
    result = await asyncio.to_thread(service.tag_tenant_tags, deps.ctx, tenant_id)
    if result.get("tenant_tags") is None:
        raise HTTPException(
            422,
            f"No tenant tags produced for tenant={tenant_id} "
            f"(no Parallel.ai profile fetched yet?).",
        )
    return result


@router.get("/api/tenants/{tenant_id}/tags")
@router.get("/api/tags")
async def get_tenant_tags(
    tenant_id: int | None = None,
    authorization: str | None = Header(default=None),
) -> dict:
    """Read tenant_tags.json."""
    tenant_id = auth.resolve_tenant_id(tenant_id, authorization)
    artifact = service.read_tenant_tags(deps.ctx, tenant_id)
    if artifact is None:
        raise HTTPException(
            404,
            f"No tenant tags for tenant={tenant_id}. "
            f"POST /api/tenants/{tenant_id}/tag to build them.",
        )
    return artifact


@router.delete("/api/tenants/{tenant_id}/tags")
@router.delete("/api/tags")
async def delete_tenant_tags(
    tenant_id: int | None = None,
    authorization: str | None = Header(default=None),
) -> dict:
    """Delete tenant_tags.json."""
    tenant_id = auth.resolve_tenant_id(tenant_id, authorization)
    result = service.delete_tenant_tags(deps.ctx, tenant_id)
    if not result["removed"]:
        raise HTTPException(404, f"No tenant tags for tenant={tenant_id}")
    return result
