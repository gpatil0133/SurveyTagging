"""Survey routes — one survey, addressed by tenant + survey number.

Lifted out of run.py unchanged — see api/__init__.py for why the split."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

import auth
import service
import sharefs
from api import deps
from projections.survey_view import build_survey_view

logger = logging.getLogger("survey_tagging.api")

router = APIRouter()


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


@router.post("/api/tenants/{tenant_id}/surveys/{survey_no}/tag")
@router.post("/api/surveys/{survey_no}/tag")
async def tag_survey(
    survey_no: int,
    tenant_id: int | None = None,
    authorization: str | None = Header(default=None),
) -> dict:
    """Tag one survey (incremental — skips if inputs unchanged)."""
    tenant_id = auth.resolve_tenant_id(tenant_id, authorization)
    try:
        return await asyncio.to_thread(service.tag_survey, deps.ctx, tenant_id, survey_no, force=False)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception:  # noqa: BLE001
        logger.exception("tag_survey_failed")
        raise deps.server_error("Tag failed")


@router.post("/api/tenants/{tenant_id}/surveys/{survey_no}/retag")
@router.post("/api/surveys/{survey_no}/retag")
async def retag_survey(
    survey_no: int,
    tenant_id: int | None = None,
    authorization: str | None = Header(default=None),
) -> dict:
    """Force re-tag one survey (ignore change detection)."""
    tenant_id = auth.resolve_tenant_id(tenant_id, authorization)
    try:
        return await asyncio.to_thread(service.tag_survey, deps.ctx, tenant_id, survey_no, force=True)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception:  # noqa: BLE001
        logger.exception("retag_survey_failed")
        raise deps.server_error("Retag failed")


@router.get("/api/tenants/{tenant_id}/surveys/{survey_no}/tags")
@router.get("/api/surveys/{survey_no}/tags")
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
    path = service.tagged_output_path(deps.ctx, tenant_id, survey_no)
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


@router.delete("/api/tenants/{tenant_id}/surveys/{survey_no}/tags")
@router.delete("/api/surveys/{survey_no}/tags")
async def delete_survey_tags(
    survey_no: int,
    tenant_id: int | None = None,
    authorization: str | None = Header(default=None),
) -> dict:
    """Delete tagged output for one survey."""
    tenant_id = auth.resolve_tenant_id(tenant_id, authorization)
    result = service.delete_tagged(deps.ctx, tenant_id, survey_no)
    if not result["tagged_removed"]:
        raise HTTPException(404, f"No tagged output for tenant={tenant_id} survey={survey_no}")
    return result


@router.post("/api/tag")
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
        return await asyncio.to_thread(service.tag_uploaded, deps.ctx, survey_json, overrides or None)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, f"Failed to parse survey structure: {e}")
