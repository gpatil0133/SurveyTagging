"""Tenant-survey routes — §1's work fanned out over every survey.

Lifted out of run.py unchanged — see api/__init__.py for why the split."""

from __future__ import annotations

import asyncio
import errno
import json
import logging

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse

import auth
import discovery
import service
from api import deps

logger = logging.getLogger("survey_tagging.api")

router = APIRouter()


# ====================================================================
# §2  Tenant surveys — §1's work fanned out over every survey
#     The two writers first (incremental, then forced), then the two readers.
#     `/tag-surveys/stream` trails `/tag-surveys` because it is the same
#     listing in a second representation, not a second listing.
# ====================================================================

@router.post("/api/tenants/{tenant_id}/tag-surveys")
@router.post("/api/tag-surveys")
async def tag_tenant_surveys(
    tenant_id: int | None = None,
    authorization: str | None = Header(default=None),
) -> dict:
    """Tag every survey under a tenant (bounded-parallel; incremental)."""
    tenant_id = auth.resolve_tenant_id(tenant_id, authorization)
    return await asyncio.to_thread(service.tag_tenant_surveys, deps.ctx, tenant_id, force=False)


@router.post("/api/tenants/{tenant_id}/retag-surveys")
@router.post("/api/retag-surveys")
async def retag_tenant_surveys(
    tenant_id: int | None = None,
    authorization: str | None = Header(default=None),
) -> dict:
    """Force re-tag every survey under a tenant."""
    tenant_id = auth.resolve_tenant_id(tenant_id, authorization)
    return await asyncio.to_thread(service.tag_tenant_surveys, deps.ctx, tenant_id, force=True)


@router.get("/api/tenants/{tenant_id}/tag-surveys")
@router.get("/api/tag-surveys")
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
        surveys = await asyncio.to_thread(service.list_survey_status, deps.ctx, tenant_id)
    except OSError:
        surveys = []          # unreadable tenant dir reads as "nothing there", as before
    if not surveys:
        raise HTTPException(404, f"No surveys on disk for tenant={tenant_id}")
    return {"tenant_id": tenant_id, "surveys": surveys}


@router.get("/api/tenants/{tenant_id}/tag-surveys/stream")
@router.get("/api/tag-surveys/stream")
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
            discovery.list_survey_dirs, deps.settings.data_dir, tenant_id
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
            async for event in service.stream_survey_status(deps.ctx, tenant_id, dirs):
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
