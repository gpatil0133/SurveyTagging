"""Build LLM prompts from survey context.

V6 changes:
- Prompt text moved to `config/prompts/*.yaml` (project_tagging, question_tagging).
  This module is now a thin adapter that gathers context and delegates rendering
  to `llm.prompt_registry.PromptRegistry`.
- Each prompt YAML carries its own `version`; the renderer returns it inside
  `RenderedPrompt`, and callers pass it to the disk response cache so bumping
  one prompt invalidates only its cache.
- `category` dimension renamed to `project_type` (taxonomy V6).

Public API:
- build_project_prompt(...)  -> RenderedPrompt
- build_question_prompt(...) -> RenderedPrompt

`RenderedPrompt.cached_preamble` is marked with cache_control=ephemeral by the
LLMClient for Anthropic prompt caching (90%+ input cost reduction on hits).
"""

from __future__ import annotations

import json
import logging

from config_loaders.industry_stages import IndustryStagesRegistry
from llm.prompt_registry import RenderedPrompt, get_registry
from models.context import UnifiedContext
from models.tags import TagAccumulator
from models.taxonomy import TaxonomyRegistry
from models.tenant_profile import TenantProfile

logger = logging.getLogger(__name__)


# ---------- Helpers (Python-side rendering of complex conditional blocks) ----------

def _build_tenant_profile_block(profile: TenantProfile | None) -> str:
    """Render Parallel.ai tenant intelligence as a hint block for LLM prompts.

    Kept in Python (rather than Jinja) because the conditional structure across
    org/CX/EX is gnarly and unit-testable here. Returns empty string when the
    profile is missing or empty so the user prompt stays compact.
    """
    if profile is None or profile.is_empty:
        return ""

    lines: list[str] = ["Tenant intelligence (from Parallel.ai research — use as a hint; survey content takes priority):"]
    if profile.has_org:
        if profile.industry_vertical:
            sub = f" / {profile.industry_sub_vertical}" if profile.industry_sub_vertical else ""
            lines.append(f"- Industry: {profile.industry_vertical}{sub}")
        if profile.regulatory_intensity or profile.data_sensitivity:
            lines.append(
                f"- Regulatory: intensity={profile.regulatory_intensity or 'Unknown'}, "
                f"data_sensitivity={profile.data_sensitivity or 'Unknown'}"
            )
        if profile.regulatory_frameworks:
            lines.append(f"- Regulatory frameworks: {', '.join(profile.regulatory_frameworks[:5])}")
    if profile.has_cx:
        if profile.primary_customer_segment:
            secondary = (
                f" (also: {', '.join(profile.secondary_customer_segments[:3])})"
                if profile.secondary_customer_segments else ""
            )
            lines.append(f"- Primary customer segment: {profile.primary_customer_segment}{secondary}")
        if profile.relationship_type:
            lines.append(
                f"- Customer relationship type: {profile.relationship_type} "
                f"(cx_confidence={profile.cx_confidence or 'Unknown'})"
            )
        if profile.customer_types:
            type_names = [str(t.get("type_name") or "?") for t in profile.customer_types[:5]]
            lines.append(f"- Customer cohorts ({len(profile.customer_types)}): {', '.join(type_names)}")
    if profile.has_ex:
        if profile.workforce_composition:
            lines.append(
                f"- Workforce: {profile.workforce_composition}"
                + (f", {profile.work_arrangement}" if profile.work_arrangement else "")
                + (f", frontline_ratio={profile.frontline_ratio}" if profile.frontline_ratio else "")
            )
        if profile.employee_types:
            type_names = [str(t.get("type_name") or "?") for t in profile.employee_types[:5]]
            lines.append(f"- Employee cohorts ({len(profile.employee_types)}): {', '.join(type_names)}")

    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def _project_cached_context(taxonomy: TaxonomyRegistry) -> dict:
    """Stable context for the project_tagging cached_preamble.

    Inputs here MUST be deterministic per-process — same taxonomy → same
    rendered string → Anthropic ephemeral cache hits.
    """
    canonical_dashboards = taxonomy.get_dimension("dashboard_routing")
    dashboard_list = canonical_dashboards.canonical_values if canonical_dashboards else []
    return {
        "relationship_types": taxonomy.get_allowed_values("relationship_type"),
        "purposes":           taxonomy.get_allowed_values("project_purpose"),
        "industries":         taxonomy.get_allowed_values("industry_vertical"),
        "audiences":          taxonomy.get_allowed_values("audience_type"),
        "sub_types":          taxonomy.get_allowed_values("survey_sub_type"),
        "canonical_dashboards": dashboard_list,
    }


def _question_cached_context(taxonomy: TaxonomyRegistry) -> dict:
    return {
        "topics":            taxonomy.get_allowed_values("topic_theme"),
        "respondent_sens":   taxonomy.get_allowed_values("respondent_sensitivity"),
        "flow_exp":          taxonomy.get_allowed_values("flow_respondent_experience"),
        "flow_reuse":        taxonomy.get_allowed_values("flow_reusability"),
        "viz":               taxonomy.get_allowed_values("visualization_type"),
        "display_roles":     taxonomy.get_allowed_values("display_role"),
    }


# ---------- Public API ----------

def build_project_prompt(
    context: UnifiedContext,
    accumulator: TagAccumulator,
    taxonomy: TaxonomyRegistry,
) -> RenderedPrompt:
    """Build project-level prompt for LLM Call 1."""
    # Gather pre-computed tags as hints (project_type renamed from V5 category)
    hint_lines = []
    for dim in ("project_type", "survey_cadence", "audience_type", "industry_vertical",
                "survey_sub_type", "dashboard_routing"):
        val = accumulator.get_project_tag_value(dim)
        if val:
            hint_lines.append(f"- {dim} (pre-computed): {val}")

    # Caller-supplied tenant hints (ad-hoc /api/tag only; empty on the disk
    # pipeline, where `tenant_profile_block` carries the tenant context).
    override_lines = []
    if context.overrides.industry:
        override_lines.append(f"Industry: {context.overrides.industry}")
    if context.overrides.company_name:
        override_lines.append(f"Company: {context.overrides.company_name}")
    if context.overrides.department:
        override_lines.append(f"Department: {context.overrides.department}")
    if context.overrides.purpose:
        override_lines.append(f"Purpose: {context.overrides.purpose}")
    if context.overrides.country:
        override_lines.append(f"Country: {context.overrides.country}")

    # Question summary (first 40)
    q_titles = []
    for q in context.questions[:40]:
        if q.is_content_message:
            q_titles.append(f"[SECTION: {q.title}]")
        else:
            prefix = ""
            if q.is_nps:
                prefix = "[NPS] "
            elif q.is_csat:
                prefix = "[CSAT] "
            elif q.is_ces:
                prefix = "[CES] "
            elif q.is_key_driver:
                prefix = "[KEY DRIVER] "
            q_titles.append(f"{prefix}{q.title}")

    directory_info = (
        ", ".join(context.directory_signals.domain_keywords[:10])
        if context.directory_signals.domain_keywords else ""
    )
    questions_block = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(q_titles))
    response_stats = (
        f"{context.response_stats.total_responses} responses over "
        f"{context.response_stats.span_days} days"
        if context.response_stats else ""
    )

    user_context = {
        "survey_title":      context.survey_meta.title,
        "survey_type":       context.survey_meta.survey_type,
        "description":       context.survey_meta.description or "",
        "overrides_info":    "\n".join(override_lines),
        "directory_info":    directory_info,
        "tenant_profile_block": _build_tenant_profile_block(context.tenant_profile),
        "has_nps":           context.has_nps,
        "has_csat":          context.has_csat,
        "total_questions":   len(context.non_cm_questions),
        "total_questions_all": len(context.questions),
        "hints_block":       "\n".join(hint_lines),
        "questions_block":   questions_block,
        "section_headers":   context.cm_section_headers if context.cm_section_headers else "",
        "response_stats":    response_stats,
    }

    return get_registry().render(
        "project_tagging",
        cached_context=_project_cached_context(taxonomy),
        user_context=user_context,
    )


def build_question_signature(context: UnifiedContext, q) -> str:
    """Compose the per-question text we score against canon stage embeddings.

    Concatenates survey-level context (title, description, section header)
    with question-level fields (matrix group, title, custom metric label,
    answer options when those add signal). The same text is also surfaced
    to the LLM as the question's `signature` so it can read what was scored.

    NPS/CSAT/CES option labels are suppressed because the 0-10 / Likert
    scale words ("Promoter", "Strongly agree") are stage-agnostic noise.
    """
    parts: list[str] = [f"Survey: {context.survey_meta.title}"]
    if getattr(context.survey_meta, "description", None):
        desc = str(context.survey_meta.description)[:200]
        if desc:
            parts.append(f"Description: {desc}")
    sec = context.section_header_for(q)
    if sec:
        parts.append(f"Section: {sec}")
    if getattr(q, "matrix_group_title", None):
        parts.append(f"Matrix group: {q.matrix_group_title}")
    parts.append(f"Question: {q.title}")
    if getattr(q, "is_custom_metric", False) and getattr(q, "custom_metric_title", None):
        parts.append(f"Custom metric: {q.custom_metric_title}")
    rs_type = getattr(q, "rs_type", 0)
    options = getattr(q, "answer_options", None) or []
    if rs_type not in (2, 3, 4) and options and len(options) <= 8:
        opts = ", ".join(o.answer_text for o in options[:6] if getattr(o, "answer_text", ""))
        if opts:
            parts.append(f"Options: {opts}")
    return " | ".join(parts)


def _question_candidates(
    context: UnifiedContext,
    q,
    top_k: int,
    canon_embeddings=None,
) -> list[tuple]:
    """Return [(CanonStage, score)] for one question, or [] if no embeddings.

    `canon_embeddings` is the journey-type-selected index (CX or EX) for this
    survey; the caller resolves it via `context.canon_for(project_type)`.
    """
    if canon_embeddings is None:
        return []
    from llm.embeddings import EmbeddingModel, score_signature
    from settings import Settings

    embedder = EmbeddingModel.get(Settings().embedding_model)
    sig = build_question_signature(context, q)
    return score_signature(sig, canon_embeddings, embedder, top_k=top_k)


def build_question_prompt(
    context: UnifiedContext,
    accumulator: TagAccumulator,
    taxonomy: TaxonomyRegistry,
    industry_stages: IndustryStagesRegistry,
    top_k: int = 4,
) -> RenderedPrompt:
    """Build question-level prompt for LLM Call 2.

    V5+: per-question candidates from canon embeddings replace the survey-wide
    industry stage list. When `context.tenant_canon` is None (legacy tenants),
    we fall back to the industry-stage-list path so the system stays runnable
    end-to-end during the transition.
    """
    industry = accumulator.get_project_tag_value("industry_vertical")
    project_type = accumulator.get_project_tag_value("project_type")
    project_dashboards = accumulator.get_project_tag_value("dashboard_routing") or []
    if not isinstance(project_dashboards, list):
        project_dashboards = [project_dashboards]

    # Select CX vs EX canon by project_type (EX surveys ground against the
    # employee-lifecycle canon).
    canon, canon_embeddings = context.canon_for(project_type)
    legacy_stage_list = (
        industry_stages.get_stages(industry, project_type=project_type)
        if canon is None else []
    )

    project_context = {
        "industry":     industry,
        "purpose":      accumulator.get_project_tag_value("project_purpose"),
        "audience":     accumulator.get_project_tag_value("audience_type"),
        "relationship": accumulator.get_project_tag_value("relationship_type"),
        "sub_type":     accumulator.get_project_tag_value("survey_sub_type"),
        "project_type": project_type,
    }

    questions_data = []
    for q in context.questions:
        if q.is_content_message:
            continue

        is_journey_metric = q.rs_type in (2, 3, 4) or bool(q.is_custom_metric)
        metric_title = q.custom_metric_title if q.is_custom_metric else None

        q_entry = {
            "id": q.question_id,
            "no": q.question_no,
            "title": q.title,
            "type": q.question_type,
            "matrix_group": q.matrix_group_title or None,
            "options": [o.answer_text for o in q.answer_options[:6] if o.answer_text],
            "current_role": accumulator.get_question_tag_value(q.question_id, "role_intent"),
            "metric_type": accumulator.get_question_tag_value(q.question_id, "metric_type"),
            "metric_name": accumulator.get_question_tag_value(q.question_id, "metric_name"),
            "trend_trackable": accumulator.get_question_tag_value(q.question_id, "trend_trackable"),
            "is_journey_metric": is_journey_metric,
            "metric_title": metric_title,
        }

        if is_journey_metric and canon is not None and canon_embeddings is not None:
            ranked = _question_candidates(context, q, top_k=top_k, canon_embeddings=canon_embeddings)
            q_entry["signature"] = build_question_signature(context, q)
            q_entry["candidates"] = [
                {
                    "stage_name": stage.name,
                    "description": stage.description,
                    "customer_goal": stage.customer_goal,
                    "synonyms": stage.synonyms[:5],
                    "score": round(score, 3),
                }
                for (stage, score) in ranked
            ]

        questions_data.append(q_entry)

    canon_note = ""
    if canon is not None:
        canon_note = (
            f'Tenant canon: "{canon.journey_name}" '
            f'({len(canon.stages)} stages, source={canon.source}). '
            f'Each journey-eligible question has its own ranked `candidates` list — '
            f'pick `journey.stage_name` EXACTLY from that list.'
        )

    user_context = {
        "survey_title":         context.survey_meta.title,
        "project_context_json": json.dumps(project_context),
        "tenant_profile_block": _build_tenant_profile_block(context.tenant_profile),
        "canon_note":           canon_note,
        "legacy_stage_list":    legacy_stage_list,
        "project_dashboards":   project_dashboards,
        "questions_count":      len(questions_data),
        "questions_json":       json.dumps(questions_data, indent=2),
    }

    return get_registry().render(
        "question_tagging",
        cached_context=_question_cached_context(taxonomy),
        user_context=user_context,
    )
