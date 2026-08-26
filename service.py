"""Use-case layer for the tagging API.

Every tagging route + the scheduler call these functions; they own the
orchestration and file I/O so the route handlers stay thin and there is one
code path per operation.

  Survey:  tag_survey / read_tagged / delete_tagged
  Listing: list_survey_status / stream_survey_status
  Tenant:  tag_tenant_surveys (all surveys, bounded-parallel)
  Tenant:  tag_tenant_tags / read_tenant_tags / delete_tenant_tags
  Ad-hoc:  tag_uploaded (in-memory survey JSON, no persistence)
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import discovery
import sharefs
from bootstrap import AppContext, build_orchestrator

logger = logging.getLogger("survey_tagging.service")


# ---------- tagged_output.json I/O ----------

def tagged_output_path(ctx: AppContext, tenant_id: int, survey_no: int) -> Path:
    return discovery.tagged_output_path(
        Path(ctx.settings.output_dir), tenant_id, survey_no
    )


def read_tagged(ctx: AppContext, tenant_id: int, survey_no: int) -> dict | None:
    path = tagged_output_path(ctx, tenant_id, survey_no)
    if not sharefs.exists(path):
        return None
    return json.loads(sharefs.read_text(path, encoding="utf-8"))


def delete_tagged(ctx: AppContext, tenant_id: int, survey_no: int | None = None) -> dict:
    """Delete tagged_output.json for one survey or every survey in a tenant.

    Deleting the artifact also drops the survey's change-detector hash. The two
    record one fact — "this survey has been tagged" — in two places, and leaving
    the hash behind makes the next tag call skip a survey whose output no longer
    exists (200 "skipped", then 404 on the read).
    """
    root = Path(ctx.settings.output_dir) / str(tenant_id)
    removed: list[str] = []
    if not sharefs.exists(root):
        return {"tenant_id": tenant_id, "survey_no": survey_no, "tagged_removed": []}

    detector = ctx.change_detector
    if survey_no is None:
        for path in sharefs.glob(root, f"SurveyData/*/{discovery.TAGGED_OUTPUT_FILE}"):
            sharefs.unlink(path, missing_ok=True)
            removed.append(str(path.relative_to(root)))
            _forget_survey_hash(detector, tenant_id, path.parent.name)
    else:
        path = tagged_output_path(ctx, tenant_id, survey_no)
        if sharefs.exists(path):
            sharefs.unlink(path)
            removed.append(str(path.relative_to(root)))
        detector.forget(tenant_id, survey_no)
    return {"tenant_id": tenant_id, "survey_no": survey_no, "tagged_removed": removed}


def _forget_survey_hash(detector, tenant_id: int, survey_dir_name: str) -> None:
    """Drop one survey's hash, given the directory name its output sat in."""
    try:
        detector.forget(tenant_id, int(survey_dir_name))
    except ValueError:  # non-numeric survey dir — nothing was ever hashed under it
        logger.debug("skip_forget_non_numeric_survey_dir | tenant=%s dir=%s",
                     tenant_id, survey_dir_name)


# ---------- survey tagging ----------

def tag_survey(ctx: AppContext, tenant_id: int, survey_no: int, *, force: bool = False) -> dict:
    """Tag one survey (full fidelity, tenant-context aware). Writes its
    tagged_output.json. `force=True` ignores change detection (retag)."""
    if not discovery.survey_exists(ctx.settings.data_dir, tenant_id, survey_no):
        raise FileNotFoundError(
            f"survey_structure.json missing for tenant={tenant_id} survey={survey_no}"
        )
    orch = build_orchestrator(ctx)
    summary = orch.run(tenant_ids=[tenant_id], survey_nos=[survey_no], force=force)
    return {
        "tenant_id": tenant_id,
        "survey_no": survey_no,
        "llm_enabled": ctx.llm_enabled,
        "summary": summary,
        "tagged": read_tagged(ctx, tenant_id, survey_no),
    }


def tag_tenant_surveys(ctx: AppContext, tenant_id: int, *, force: bool = False) -> dict:
    """Tag every survey under a tenant (bounded-parallel per settings). Also
    refreshes tenant_tags.json as part of the orchestrator run."""
    orch = build_orchestrator(ctx)
    summary = orch.run(tenant_ids=[tenant_id], survey_nos=None, force=force)
    tenant_summary = summary.get("tenant_summaries", {}).get(tenant_id, {})
    return {
        "tenant_id": tenant_id,
        "llm_enabled": ctx.llm_enabled,
        "processed": summary.get("total_surveys_processed", 0),
        "skipped": summary.get("total_surveys_skipped", 0),
        "failed": summary.get("total_surveys_failed", 0),
        "surveys": tenant_summary.get("surveys", []),
    }


# ---------- survey listing (tagged status) ----------
#
# A tenant's listing is one directory listing plus one probe per survey, and each
# probe is a network round trip. At a few thousand surveys the serial form takes
# minutes and the browser (or a proxy in front of it) gives up before the first
# byte, so both listing paths here fan the probes out across a thread pool: the
# threads share one authenticated SMB session safely (see sharefs' module
# docstring) and the round trips overlap, up to the server's credit window —
# which is what `discovery_workers` is really sized against, see `_probe`.
#
# `stream_survey_status` additionally yields each survey the moment its probe
# lands, which is what keeps the response alive: bytes flow continuously, so no
# timeout is waiting on a total that only exists at the end.

_PING_SECONDS = 10.0     # keepalive cadence while every probe is still in flight
_PROBE_ATTEMPTS = 3


def _probe(ctx: AppContext, tenant_id: int, sdir: Path) -> dict | None:
    """One survey's status, with a retry for the failure the fan-out causes itself.

    SMB2 hands the client a bounded number of *credits* — outstanding requests
    the server will accept — and smbprotocol raises `SMBException: Request
    requires 1 credits but only 0 credits are available` the moment the pool
    outruns that window. Measured against the QA image server the wall is
    somewhere between 8 and 12 concurrent probes, and it is a property of the
    server, not of this code: a different share grants a different window, so no
    fixed `discovery_workers` can be guaranteed safe.

    It is also transient by construction — the credits come back as the in-flight
    requests return — so a short jittered wait and a retry clears it. What must
    not happen is the alternative: an exception here kills the whole listing, and
    an exception swallowed here silently drops a survey the user owns. After the
    last attempt the survey is still reported, with `tagged: None` for "could not
    tell", which the UI already renders as an unknown-state row.
    """
    sno = int(sdir.name)
    data_dir, output_dir = Path(ctx.settings.data_dir), Path(ctx.settings.output_dir)
    for attempt in range(_PROBE_ATTEMPTS):
        try:
            return discovery.probe_survey(data_dir, output_dir, tenant_id, sno)
        except Exception as e:  # noqa: BLE001 — smbprotocol raises SMBException, not OSError
            if attempt < _PROBE_ATTEMPTS - 1:
                time.sleep(random.uniform(0.1, 0.4) * (attempt + 1))
                continue
            logger.warning("survey_probe_failed", extra={
                "tenant_id": tenant_id, "survey_no": sno,
                "error": f"{type(e).__name__}: {e}"})
            return {"survey_no": sno, "tagged": None, "probe_error": str(e)}
    return None


def _workers(ctx: AppContext, n: int) -> int:
    return max(1, min(getattr(ctx.settings, "discovery_workers", 4), n))


def list_survey_status(ctx: AppContext, tenant_id: int) -> list[dict]:
    """`[{survey_no, tagged}]` for every survey under a tenant, sorted.

    Raises OSError when the tenant's SurveyData dir cannot be listed.
    """
    dirs = discovery.list_survey_dirs(Path(ctx.settings.data_dir), tenant_id)
    if not dirs:
        return []
    with ThreadPoolExecutor(max_workers=_workers(ctx, len(dirs))) as pool:
        rows = pool.map(lambda d: _probe(ctx, tenant_id, d), dirs)
    return sorted((r for r in rows if r), key=lambda r: r["survey_no"])


async def stream_survey_status(
    ctx: AppContext, tenant_id: int, dirs: list[Path]
) -> AsyncIterator[dict]:
    """Yield listing events for a tenant, one per survey as its probe lands.

    Events: `{kind: "start", tenant_id, scanning}`, then a `{kind: "survey",
    survey_no, tagged}` per survey **in completion order, not sorted** (the
    consumer orders them — waiting for order would reintroduce the stall this
    exists to remove), a `{kind: "ping"}` whenever a whole cadence passes with
    nothing finished, and finally `{kind: "done", count}`.

    `dirs` is passed in rather than listed here so the route can still answer a
    dead share with an HTTP error: once the first byte is out the status code is
    already spent.
    """
    yield {"kind": "start", "tenant_id": tenant_id, "scanning": len(dirs)}
    if not dirs:
        yield {"kind": "done", "count": 0}
        return

    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(max_workers=_workers(ctx, len(dirs)),
                              thread_name_prefix="survey-probe")
    count = 0
    try:
        pending = {loop.run_in_executor(pool, _probe, ctx, tenant_id, d) for d in dirs}
        while pending:
            done, pending = await asyncio.wait(
                pending, timeout=_PING_SECONDS, return_when=asyncio.FIRST_COMPLETED
            )
            if not done:
                # Nothing landed this cadence. Say so rather than going quiet:
                # silence on the wire is indistinguishable from a hung server
                # and is what the client's idle timer would shoot.
                yield {"kind": "ping"}
                continue
            for fut in done:
                try:
                    row = fut.result()
                except Exception as e:    # noqa: BLE001 — _probe already retries; a
                    # survivor here must still not take the rest of the listing down.
                    logger.warning("survey_probe_failed", extra={"tenant_id": tenant_id,
                                                                 "error": str(e)})
                    continue
                if row:
                    count += 1
                    yield {"kind": "survey", **row}
        yield {"kind": "done", "count": count}
    finally:
        # The client can disconnect mid-listing (closed tab, switched corp).
        # Don't wait for the queued probes — drop what has not started.
        pool.shutdown(wait=False, cancel_futures=True)


# ---------- tenant-level tags (tenant_tags.json) ----------

def tag_tenant_tags(ctx: AppContext, tenant_id: int) -> dict:
    """Build + persist tenant_tags.json (tenant taggers + Parallel.ai profile)."""
    orch = build_orchestrator(ctx)
    artifact = orch.tag_tenant_only(tenant_id)
    return {"tenant_id": tenant_id, "tenant_tags": artifact}


def read_tenant_tags(ctx: AppContext, tenant_id: int) -> dict | None:
    from projections.tenant_tags_io import load_tenant_tags
    artifact = load_tenant_tags(tenant_id, Path(ctx.settings.output_dir))
    return artifact.model_dump() if artifact else None


def delete_tenant_tags(ctx: AppContext, tenant_id: int) -> dict:
    from projections.tenant_tags_io import tenant_tags_path
    path = tenant_tags_path(tenant_id, Path(ctx.settings.output_dir))
    existed = sharefs.exists(path)
    if existed:
        sharefs.unlink(path)
    # Same pairing as delete_tagged: the artifact and its hash go together.
    ctx.change_detector.tenant_forget(tenant_id)
    return {"tenant_id": tenant_id, "removed": existed}


# ---------- ad-hoc upload (no tenant on disk, no persistence) ----------

def tag_uploaded(ctx: AppContext, survey_json: dict, overrides: dict | None = None) -> dict:
    """Tag an in-memory survey JSON via the per-survey engine. Deterministic —
    no tenant context, canon, or LLM (there is no tenant on disk to ground it).

    `overrides` carries the caller's manual tenant hints (industry,
    company_name, department, purpose, country)."""
    from loaders.context_assembler import assemble_context_from_json
    from pipeline.single_survey import process_single_survey

    context = assemble_context_from_json(survey_json, overrides or None)
    result = process_single_survey(context, ctx.registry, ctx.taxonomy)
    return result.model_dump()
