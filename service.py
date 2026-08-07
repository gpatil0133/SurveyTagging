"""Use-case layer for the tagging API.

Every tagging route + the scheduler call these functions; they own the
orchestration and file I/O so the route handlers stay thin and there is one
code path per operation.

  Survey:  tag_survey / read_tagged / delete_tagged
  Tenant:  tag_tenant_surveys (all surveys, bounded-parallel)
  Tenant:  tag_tenant_tags / read_tenant_tags / delete_tenant_tags
  Ad-hoc:  tag_uploaded (in-memory survey JSON, no persistence)
"""

from __future__ import annotations

import json
import logging
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
    """Delete tagged_output.json for one survey or every survey in a tenant."""
    root = Path(ctx.settings.output_dir) / str(tenant_id)
    removed: list[str] = []
    if not sharefs.exists(root):
        return {"tenant_id": tenant_id, "survey_no": survey_no, "tagged_removed": []}

    if survey_no is None:
        for path in sharefs.glob(root, f"SurveyData/*/{discovery.TAGGED_OUTPUT_FILE}"):
            sharefs.unlink(path, missing_ok=True)
            removed.append(str(path.relative_to(root)))
    else:
        path = tagged_output_path(ctx, tenant_id, survey_no)
        if sharefs.exists(path):
            sharefs.unlink(path)
            removed.append(str(path.relative_to(root)))
    return {"tenant_id": tenant_id, "survey_no": survey_no, "tagged_removed": removed}


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
