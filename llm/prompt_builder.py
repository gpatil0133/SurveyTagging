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
    "relationship_type", "project_purpose", "project_intent", "industry_vertical",
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
        # No `industries`: industry_vertical is user_defined free text, so its
        # allowed_values list is empty and the prompt asks for prose instead.
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
    # V8. All three follow the same rule as the rest — a settled tag wins the
    # merge, so asking again buys nothing — and all three are settled far more
    # often than not: `favorable_options` only reaches the model on an
    # UNWEIGHTED scale, `display_label` only when the question has no metric
    # name, and `preferred_segments` only when a metric has candidates to rank.
    # The taggers mark exactly those cases `pending_llm`, which `_is_settled`
    # reads as unsettled.
    "favorable_options":          ("favorable_options", 0.80),
    "preferred_segments":         ("preferred_segments", 0.80),
    "display_label":              ("display_label", 0.80),
}

# Output field -> the per-question input the model needs to answer it. Asking
# for one of these without supplying its input would invite the model to invent
# ids, so the two are gated together in `_needed_fields`.
_FIELD_REQUIRES_INPUT = {
    "favorable_options": "options_for_favorability",
    "preferred_segments": "segment_candidates",
}

# Roles whose options are alternatives rather than points on a good-to-bad
# scale, so there is no favorability to split. The tagger cannot rule these out
# itself — it runs at Stage 3, before `role_intent` — but by the time a prompt
# is built the role is known, and "Which region are you in?" is a list where no
# option is favorable. Skipping them here drops the option payload AND the
# request from `needed`, on what is otherwise the most common shape in a survey.
_NEVER_FAVORABILITY_ROLES = frozenset({
    "Segmentation", "Profiling / Demographic", "Screener",
})


def _needed_fields(
    accumulator: TagAccumulator,
    q_id: int,
    *,
    wants_journey: bool,
    available_inputs: set[str] | None = None,
) -> list[str]:
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

    `available_inputs` names the per-question inputs the caller actually
    attached. A field in `_FIELD_REQUIRES_INPUT` is dropped when its input is
    missing, because asking a model to pick from a list it was not shown is an
    invitation to invent ids. Defaults to none attached, which is the safe
    direction for a caller that does not build the payload.
    """
    supplied = (available_inputs or set()) | {""}
    needed = [
        field
        for field, (dimension, threshold) in _MASKABLE_FIELDS.items()
        if not _is_settled(accumulator.get_question_tag(q_id, dimension), threshold)
        and _FIELD_REQUIRES_INPUT.get(field, "") in supplied
    ]
    # Always requested: the merge rules apply these no matter what ran first.
    needed.append("dashboard_names")
    # Always requested, but nearly free: the field is a union onto whatever the
    # structural pass found, so no confidence threshold can settle it — a
    # question can be a known Branching Target AND an unlisted Branching Trigger.
    # The prompt tells the model to OMIT the key unless it actually infers a
    # role, and on measured surveys that is ~99% of questions.
    needed.append("flow_logic_inferred")
    if wants_journey:
        needed.append("journey")
    return needed


def _awaiting_llm(accumulator: TagAccumulator, q_id: int, dimension: str) -> bool:
    """True when a tagger deliberately left this dimension for LLM Call 2.

    The V8 taggers mark exactly the cases a rule cannot decide `pending_llm` —
    an unweighted scale for `favorable_options`, a metric with candidates for
    `preferred_segments` — so this is what gates attaching their (non-trivial)
    prompt inputs. Anything settled, skipped, or answered by a rule sends no
    extra payload at all.
    """
    existing = accumulator.get_question_tag(q_id, dimension)
    return existing is not None and existing.status == "pending_llm"


def _is_settled(existing, threshold: float) -> bool:
    """True when the prior tag will beat anything the LLM returns.

    Mirrors the merge rules' own test — `TagResult.confidence` is a required
    float there too, so a bare comparison is the honest shape.
    """
    if existing is None:
        return False
    return existing.status == "skipped" or existing.confidence >= threshold


def build_question_prompt(
    context: UnifiedContext,
    accumulator: TagAccumulator,
    taxonomy: TaxonomyRegistry,
    questions: list | None = None,
) -> RenderedPrompt:
    """Build question-level prompt for LLM Call 2.

    `questions` renders a subset (one batch) instead of the whole survey.

    V9: the tenant's journey is inlined **whole**, once per prompt, as a
    `journey.moments` catalog the model selects a `leaf_id` from — replacing the
    per-question top-4 an embedding index used to rank. Two reasons, both
    against the old shape: the catalog is sent once where the ranked lists were
    repeated per question (so this is fewer input tokens, not more, on any
    survey with more journey-eligible questions than the old `top_k` of 4), and
    a leaf the ranker put below the cut was unrecoverable — the model never saw
    it. Selection quality is now bounded by the model rather than by a 384-dim
    encoder scoring against a synthesized question signature.

    V8: the journey comes from the tenant profile (`context.journey_for`). There
    is no industry-template fallback — a tenant with no profile gets no journey
    catalog, and the questions are marked skipped with evidence rather than
    assigned a generic stage.
    """
    industry = accumulator.get_project_tag_value("industry_vertical")
    project_type = accumulator.get_project_tag_value("project_type")
    project_dashboards = accumulator.get_project_tag_value("dashboard_routing") or []
    if not isinstance(project_dashboards, list):
        project_dashboards = [project_dashboards]

    # Select CX vs EX journey by project_type (EX surveys ground against the
    # employee lifecycle).
    journey = context.journey_for(project_type)

    project_context = {
        "industry":     industry,
        "purpose":      accumulator.get_project_tag_value("project_purpose"),
        "audience":     accumulator.get_project_tag_value("audience_type"),
        "relationship": accumulator.get_project_tag_value("relationship_type"),
        "sub_type":     accumulator.get_project_tag_value("survey_sub_type"),
        "project_type": project_type,
    }

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
            # The nearest preceding section header. It used to reach the model
            # only inside the embedding `signature`, which went away with the
            # ranker — but the journey rules lean on it explicitly to place a
            # generically-worded metric ("Please rate your level of agreement"),
            # so it is a first-class field now rather than a casualty.
            "section": context.section_header_for(q) or None,
            "options": [o.answer_text for o in q.answer_options[:6] if o.answer_text],
            "current_role": accumulator.get_question_tag_value(q.question_id, "role_intent"),
            # Flow context for `flow_logic_inferred`. The payload carries no
            # skip-logic definitions, so these four platform flags plus the
            # question's own position are the entire structural picture — the
            # model has to be shown what IS known or it will re-assert it.
            "position_pct": round(q.effective_position_ratio * 100),
            "is_followup": bool(q.is_followup_question),
            "flow_logic_detected": accumulator.get_question_tag_value(
                q.question_id, "flow_logic_role") or [],
            "metric_type": accumulator.get_question_tag_value(q.question_id, "metric_type"),
            "metric_name": accumulator.get_question_tag_value(q.question_id, "metric_name"),
            "trend_trackable": accumulator.get_question_tag_value(q.question_id, "trend_trackable"),
            "is_journey_metric": is_journey_metric,
            "metric_title": metric_title,
        }

        # Whether to ask this question for a journey placement at all. The
        # catalog is survey-wide, so this is the only per-question gate left:
        # a non-metric question has no moment to be measuring, and a tenant with
        # no journey has nothing to offer.
        wants_journey = is_journey_metric and journey is not None

        # V8 inputs, attached only when the matching tag is still open. Both are
        # id lists the model picks FROM rather than writes, the same shape as
        # the journey catalog — nothing the model types becomes a tag value,
        # so it cannot invent an option or a question that does not exist.
        available_inputs: set[str] = set()

        if (_awaiting_llm(accumulator, q.question_id, "favorable_options")
                and q_entry["current_role"] not in _NEVER_FAVORABILITY_ROLES):
            # The full option list with ids, not the six-title preview above:
            # a favorability split has to cover every option exactly once.
            options = [{"id": o.answer_id, "text": o.answer_text}
                       for o in q.answer_options if o.answer_text]
            if options:
                q_entry["options_for_favorability"] = options
                available_inputs.add("options_for_favorability")

        if _awaiting_llm(accumulator, q.question_id, "preferred_segments"):
            candidates = [
                {"id": other.question_id, "title": other.title}
                for other in context.questions
                if not other.is_content_message
                and other.question_id != q.question_id
                and accumulator.get_question_tag_value(
                    other.question_id, "is_segmentable") == "Yes"
            ]
            if candidates:
                q_entry["segment_candidates"] = candidates
                available_inputs.add("segment_candidates")

        # Ask only for what the merge rules will actually keep.
        q_entry["needed"] = _needed_fields(
            accumulator, q.question_id, wants_journey=wants_journey,
            available_inputs=available_inputs,
        )

        questions_data.append(q_entry)

    # The journey catalog: every moment the tenant has, once, for the whole
    # batch. Rendered as its own block rather than per question — that is the
    # entire token argument for dropping the ranker, and duplicating it per
    # question would give the cost back.
    journey_block = ""
    if journey is not None:
        sub_stage_rule = (
            "Each moment carries both levels: `stage_name` is the journey it "
            "belongs to and `sub_stage_name` is the specific moment within it. "
            "Several moments share a `stage_name` — that is why you return "
            "`leaf_id` and never type a stage name."
            if journey.has_sub_stages else
            "This journey has one level only: moments carry `stage_name` with no "
            "`sub_stage_name`, and no sub-stage will be recorded."
        )
        journey_block = (
            f'Tenant journey: "{journey.journey_name}" — '
            f'{len(journey.leaves)} moments across {len(journey.stage_values)} '
            f'stages, read from the tenant profile. {sub_stage_rule}\n'
            f'{json.dumps(journey.catalog(), indent=2)}'
        )

    user_context = {
        "survey_title":         context.survey_meta.title,
        "project_context_json": json.dumps(project_context),
        "tenant_profile_block": _build_tenant_profile_block(context.tenant_profile),
        "journey_block":        journey_block,
        "project_dashboards":   project_dashboards,
        "questions_count":      len(questions_data),
        "questions_json":       json.dumps(questions_data, indent=2),
    }

    return get_registry().render(
        "question_tagging",
        cached_context=_question_cached_context(taxonomy),
        user_context=user_context,
    )
