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
- build_question_prompt(...) -> (RenderedPrompt, candidates_by_qid)

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


# Project dimensions the LLM assigns, in the order the guide renders them.
_PROJECT_GUIDE_DIMS = (
    "relationship_type", "project_purpose", "industry_vertical",
    "audience_type", "survey_sub_type", "dashboard_routing",
)


def _dimension_guide(taxonomy: TaxonomyRegistry, dimensions) -> list[dict]:
    """Plain-language meaning of each dimension, sourced from the taxonomy.

    The schema block lists bare enum names; this says what the dimension *is*,
    so the model is choosing between described concepts rather than guessing
    from labels. Sourced from `taxonomy.yaml` rather than restated here so the
    prompt and the docs cannot drift apart.

    Stable per taxonomy, so it belongs in the cached preamble — where it also
    lifts that preamble over the providers' 1024-token minimum cacheable
    prefix. Below that, `cache_control` is silently ignored.
    """
    guide = []
    for name in dimensions:
        dim = taxonomy.get_dimension(name)
        if dim is None:
            continue
        text = (dim.explanation or dim.description or "").strip()
        if text:
            guide.append({"name": name, "text": text})
    return guide


def _project_cached_context(taxonomy: TaxonomyRegistry) -> dict:
    """Stable context for the project_tagging cached_preamble.

    Inputs here MUST be deterministic per-process — same taxonomy → same
    rendered string → prompt-cache hits.
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
        "dimension_guide":    _dimension_guide(taxonomy, _PROJECT_GUIDE_DIMS),
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
    current_section = ""
    for q in context.questions[:40]:
        # Section headers come from the content messages the loader folded into
        # `section_header`; emit one line each time the section changes.
        if q.section_header and q.section_header != current_section:
            current_section = q.section_header
            q_titles.append(f"[SECTION: {current_section}]")

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


# Output field -> (accumulator dimension, confidence at or above which
# `_apply_question_llm_results` discards the model's answer).
#
# These mirror the merge rules exactly. `dashboard_placement` and the journey
# block are deliberately absent: both are written regardless of the prior tag's
# confidence, so their answers are never wasted and must always be requested.
_MASKABLE_FIELDS: dict[str, tuple[str, float]] = {
    "topic_theme":                ("topic_theme", 0.80),
    "respondent_sensitivity":     ("respondent_sensitivity", 0.80),
    "flow_respondent_experience": ("flow_respondent_experience", 0.80),
    "flow_reusability":           ("flow_reusability", 0.80),
    "visualization_type":         ("visualization_type", 0.80),
    "display_role":               ("display_role", 0.80),
    "role_intent_refined":        ("role_intent", 0.70),
}


def _needed_fields(accumulator: TagAccumulator, q_id: int, *, has_candidates: bool) -> list[str]:
    """The output fields still worth asking the model for on this question.

    A deterministic tagger that already answered with enough confidence wins the
    merge, so requesting that field again buys nothing and costs both the value
    and its `why` line on every question. Measured across the sample corpus,
    `role_intent` is already settled 95% of the time and `display_role` 79%.

    This is purely an economy: omitting a field makes the parser take the same
    `if not value: continue` branch it already takes when the merge rules drop
    the answer, so the resulting tags are identical either way.

    Order is fixed (dict insertion order) so the rendered prompt stays stable
    for identical input — a shuffled list would be a silent cache invalidator.
    """
    needed = [
        field
        for field, (dimension, threshold) in _MASKABLE_FIELDS.items()
        if not _is_settled(accumulator.get_question_tag(q_id, dimension), threshold)
    ]
    # Always requested: the merge rules apply these no matter what ran first.
    needed.append("dashboard_names")
    if has_candidates:
        needed.append("journey")
    return needed


def _is_settled(existing, threshold: float) -> bool:
    """True when the prior tag will beat anything the LLM returns.

    Mirrors the merge rules' own test — `TagResult.confidence` is a required
    float there too, so a bare comparison is the honest shape.
    """
    if existing is None:
        return False
    return existing.status == "skipped" or existing.confidence >= threshold


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


def build_question_candidates(
    context: UnifiedContext,
    journey_index,
    *,
    top_k: int,
    embedding_model: str,
    min_score: float = 0.0,
) -> dict[int, list[dict]]:
    """Score every journey-eligible question against the journey in ONE encode pass.

    Returns `{question_id: [candidate, ...]}` where each candidate carries the
    fields the LLM prompt shows *and* the `leaf_id` the response parser resolves
    the model's pick through — one shape, one computation, two consumers.

    This is deliberately survey-wide rather than per-question: `score_questions`
    batches the transformer forward pass, and the previous per-question call ran
    it once per question, twice per survey (prompt build + parser gather).

    `journey_index` is the journey-type-selected index (CX or EX) for this
    survey; the caller resolves it via `context.journey_for(project_type)`.

    A question whose best leaf scores below `min_score` is **omitted entirely**.
    That is the honest outcome for a metric the tenant's journey has no home for
    — the alternative is forcing it into the nearest stage and reporting a match
    that isn't one. The caller distinguishes "below floor" from "no journey at
    all" and records the difference in the tag's evidence.
    """
    if journey_index is None:
        return {}
    try:
        from llm.profile_journey import score_questions
        from llm.embeddings import EmbeddingModel
        from taggers._metric_utils import is_journey_eligible_metric

        eligible = [
            q for q in context.questions
            if not q.is_content_message and is_journey_eligible_metric(q)[0]
        ]
        if not eligible:
            return {}

        embedder = EmbeddingModel.get(embedding_model)
        signatures = [build_question_signature(context, q) for q in eligible]
        ranked = score_questions(signatures, journey_index, embedder, top_k=top_k)

        out: dict[int, list[dict]] = {}
        for q, r in zip(eligible, ranked):
            if not r or r[0][1] < min_score:
                continue
            out[q.question_id] = [
                {
                    "leaf_id": leaf.leaf_id,
                    "stage_name": leaf.stage_value,
                    "sub_stage_name": leaf.sub_stage_value,
                    "description": leaf.description,
                    "goal": leaf.goal,
                    "score": round(score, 3),
                }
                for (leaf, score) in r
            ]
        return out
    except Exception as e:  # noqa: BLE001 — scoring is an enhancement, never fatal
        logger.warning("question_candidate_scoring_failed", extra={"error": str(e)})
        return {}


def build_question_prompt(
    context: UnifiedContext,
    accumulator: TagAccumulator,
    taxonomy: TaxonomyRegistry,
    industry_stages: IndustryStagesRegistry = None,
    top_k: int = 4,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    min_score: float = 0.0,
    questions: list | None = None,
    candidates_by_qid: dict[int, list[dict]] | None = None,
) -> tuple[RenderedPrompt, dict[int, list[dict]]]:
    """Build question-level prompt for LLM Call 2.

    Returns `(rendered_prompt, candidates_by_qid)`. The candidate map is
    returned rather than recomputed downstream because the response parser
    resolves the model's pick through the identical ranking — deriving it twice
    cost a second full embedding pass and let the two copies disagree whenever
    `top_k` differed between the call sites.

    `questions` renders a subset (one batch) instead of the whole survey, and
    `candidates_by_qid` supplies a ranking already computed for the full survey.
    Batched callers must pass both: scoring is survey-wide and batching it would
    reinstate the per-call embedding pass this signature exists to avoid.

    `top_k`, `embedding_model` and `min_score` are passed in (rather than read
    from a freshly-constructed `Settings()`) so a single survey does not
    re-parse `.env` from disk once per question.

    V8: journey candidates come from the tenant profile (`context.journey_for`).
    There is no industry-template fallback — a tenant with no profile gets no
    journey candidates, and the questions are marked skipped with evidence
    rather than assigned a generic stage. `industry_stages` is retained in the
    signature for callers built against the parked canon path; it is unused.
    """
    industry = accumulator.get_project_tag_value("industry_vertical")
    project_type = accumulator.get_project_tag_value("project_type")
    project_dashboards = accumulator.get_project_tag_value("dashboard_routing") or []
    if not isinstance(project_dashboards, list):
        project_dashboards = [project_dashboards]

    # Select CX vs EX journey by project_type (EX surveys ground against the
    # employee lifecycle).
    journey, journey_index = context.journey_for(project_type)

    project_context = {
        "industry":     industry,
        "purpose":      accumulator.get_project_tag_value("project_purpose"),
        "audience":     accumulator.get_project_tag_value("audience_type"),
        "relationship": accumulator.get_project_tag_value("relationship_type"),
        "sub_type":     accumulator.get_project_tag_value("survey_sub_type"),
        "project_type": project_type,
    }

    # One embedding pass for the whole survey; reused by the prompt below and
    # returned to the caller for the parser. A batched caller scores once and
    # passes the map back in, so this runs a single time per survey.
    if candidates_by_qid is None:
        candidates_by_qid = build_question_candidates(
            context, journey_index, top_k=top_k, embedding_model=embedding_model,
            min_score=min_score,
        )

    questions_data = []
    for q in (questions if questions is not None else context.questions):
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

        ranked = candidates_by_qid.get(q.question_id) if (
            is_journey_metric and journey is not None
        ) else None
        if ranked:
            q_entry["signature"] = build_question_signature(context, q)
            q_entry["candidates"] = ranked

        # Ask only for what the merge rules will actually keep.
        q_entry["needed"] = _needed_fields(
            accumulator, q.question_id, has_candidates=bool(ranked),
        )

        questions_data.append(q_entry)

    journey_note = ""
    if journey is not None:
        sub_stage_rule = (
            "Each candidate carries both levels: `stage_name` is the journey the "
            "moment belongs to and `sub_stage_name` is the specific moment within "
            "it. Return the candidate's `leaf_id` — both tag values are resolved "
            "from it, so you never type a stage name."
            if journey.has_sub_stages else
            "This journey has one level only: candidates carry `stage_name` with a "
            "null `sub_stage_name`, and no sub-stage will be recorded. Return the "
            "candidate's `leaf_id`."
        )
        journey_note = (
            f'Tenant journey: "{journey.journey_name}" '
            f'({len(journey.leaves)} moments across {len(journey.stage_values)} '
            f'stages, read from the tenant profile). {sub_stage_rule} '
            f'A question with no `candidates` list has no home in this journey — '
            f'set its `journey` to null rather than guessing.'
        )

    user_context = {
        "survey_title":         context.survey_meta.title,
        "project_context_json": json.dumps(project_context),
        "tenant_profile_block": _build_tenant_profile_block(context.tenant_profile),
        "journey_note":         journey_note,
        "project_dashboards":   project_dashboards,
        "questions_count":      len(questions_data),
        "questions_json":       json.dumps(questions_data, indent=2),
    }

    rendered = get_registry().render(
        "question_tagging",
        cached_context=_question_cached_context(taxonomy),
        user_context=user_context,
    )
    return rendered, candidates_by_qid
