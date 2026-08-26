"""LLM enhancement for a single survey (Stage 4 project + Stage 5 question).

Extracted from the orchestrator so the per-survey engine and the tenant
orchestrator share one implementation. Pure function over a UnifiedContext +
TagAccumulator; mutates the accumulator in place and returns the number of LLM
calls made.
"""

from __future__ import annotations

import asyncio
import json
import logging

from llm.cache import LLMCache
from llm.prompt_builder import build_project_prompt, build_question_prompt
from models import evidence as ev
from models.tags import TagResult

logger = logging.getLogger(__name__)

# LLM confidence string → numeric for accumulator merge ranking.
_CONF_NUMERIC = {"high": 0.85, "medium": 0.75, "low": 0.55, "none": 0.40}

# Confidence stamped on an LLM-assigned tag. Project-level reads the whole survey
# and lands slightly higher than a per-question judgement; both sit below the
# deterministic floor so a later rule-based pass would still win.
_PROJECT_CONFIDENCE = 0.85
_QUESTION_CONFIDENCE = 0.80


def _rationale(why: dict, field: str, summary: str | None) -> str | None:
    """The explanation stamped on ONE dimension's tag.

    Prefers the model's per-dimension `why` line; falls back to the
    question/survey summary so a tag is never left unexplained — including for
    responses replayed from the disk cache under prompt version < 7.1, which
    predate `why` entirely.
    """
    line = why.get(field) if isinstance(why, dict) else None
    if isinstance(line, str) and line.strip():
        return line.strip()
    return summary or None


def _journey_fingerprint(journey) -> dict:
    """Identity of the journey a question prompt was grounded in.

    Two journeys with the same leaves in the same order produce the same
    candidates and may safely share a cache entry; anything else must not — a
    refreshed tenant profile has to invalidate the response.
    """
    if journey is None:
        return {"journey": None}
    return {
        "journey": journey.journey_name,
        "journey_hash": journey.source_hash,
        "journey_leaves": [leaf.leaf_id for leaf in journey.leaves],
    }


def _batch(items: list, size: int) -> list[list]:
    """Split `items` into consecutive batches of at most `size`.

    A fixed cap rather than a token budget: the split for a given survey must be
    a function of the question list alone. Sizing batches from estimated output
    would move every boundary whenever tagger confidence shifted, and each moved
    boundary is a silent cache miss for that batch and all after it.

    Order is preserved so a question's neighbours — and the section context they
    carry — stay with it.
    """
    if size < 1:
        size = 1
    return [items[i:i + size] for i in range(0, len(items), size)] or [[]]


def _cache_key(parts: dict) -> str:
    """Hash the full set of inputs a cached response actually depends on.

    `sort_keys` matters: dict ordering must not decide whether two identical
    surveys share an entry.
    """
    return LLMCache.compute_hash(json.dumps(parts, sort_keys=True, default=str))


def _superseded(existing) -> dict | None:
    """Snapshot the tag an LLM answer is about to replace, so the override is
    visible in the output instead of erasing the rule that ran first.

    Returns None when there is nothing meaningful to record (no prior tag, no
    prior value, or the prior tag was itself from the LLM).
    """
    if existing is None or existing.value is None or existing.source == "llm":
        return None
    entry: dict = {
        "value": existing.value,
        "source": existing.source,
        "confidence": existing.confidence,
    }
    if existing.evidence:
        entry["evidence"] = existing.evidence
    return entry


def run_llm_enhancement(
    context,
    accumulator,
    *,
    llm_client,
    taxonomy,
    response_parser,
    settings,
) -> int:
    """Run LLM calls for semantic tagging. Returns number of calls made."""
    calls_made = 0

    # The survey's own identity. `tenant_id` is part of it because the response
    # is tenant-shaped — the prompts carry the tenant profile, and the question
    # prompt carries that tenant's canon. Without it two tenants running the
    # same survey template (the common case for a standard engagement survey)
    # share one flat cache entry and inherit each other's journey stages.
    survey_identity = {
        "tenant": context.tenant_id,
        "title": context.survey_meta.title,
        "questions": [q.title for q in context.questions],
    }
    project_key = _cache_key(survey_identity)
    logger.debug("llm_cache_key_computed", extra={"project_key": project_key})

    # One loop for the whole survey, closed in `finally`. It cannot be one loop
    # per call: `LLMClient._semaphore` binds to the first loop that awaits it and
    # raises on any other, which is also why parallel workers each hold their own
    # client. Closed unconditionally — a raise anywhere below used to leak the
    # loop and its selector, once per failed survey, for the life of the process.
    loop = asyncio.new_event_loop()
    try:
        # LLM Call 1: Project-level
        logger.debug("llm_project_call_start", extra={"cache_key": project_key})
        rp = build_project_prompt(context, accumulator, taxonomy)
        logger.debug("project_prompt_built",
                     extra={"prompt_version": rp.version,
                            "user_prompt_chars": len(rp.user_prompt),
                            "cached_preamble_chars": len(rp.cached_preamble or "")})
        result = loop.run_until_complete(
            llm_client.complete(
                rp.user_prompt, "",
                cache_key=project_key, call_type="project",
                cached_system_preamble=rp.cached_preamble,
                prompt_version=rp.version,
            )
        )
        if result and response_parser:
            parsed = response_parser.parse_project_response(result)
            _apply_project_llm_results(parsed, accumulator)
            calls_made += 1
            logger.debug("llm_project_call_applied", extra={"cache_key": project_key})
        else:
            logger.debug("llm_project_call_no_result",
                         extra={"cache_key": project_key,
                                "has_result": result is not None})

        # LLM Call 2: Question-level. Deliberately not gated on any journey
        # source: this call also assigns seven dimensions that have nothing to do
        # with journeys, and a tenant with no profile must not lose topic_theme,
        # visualization_type and the rest along with its journey stages.
        if context.non_cm_questions:
            industry = accumulator.get_project_tag_value("industry_vertical")
            project_type = accumulator.get_project_tag_value("project_type")
            # Select CX vs EX journey by project_type (EX surveys ground against
            # the employee lifecycle). Resolved before the call because the
            # journey is a cache-key input, not just a prompt input.
            journey = context.journey_for(project_type)

            # The question response is only valid for the journey that was
            # inlined into its prompt — a refreshed tenant profile must
            # invalidate it. V9 dropped `top_k` / `min_score` from the key with
            # the ranker they configured; the journey fingerprint already covers
            # every leaf the model was shown.
            base_key = {**survey_identity, **_journey_fingerprint(journey)}

            batches = _batch(context.non_cm_questions,
                             int(getattr(settings, "question_batch_size", 20) or 20))
            logger.debug("llm_question_call_start",
                         extra={"non_cm_questions": len(context.non_cm_questions),
                                "batches": len(batches)})

            parsed_all: list[dict] = []
            for index, batch in enumerate(batches):
                # The batch's own question ids are part of its identity — without
                # them every batch of a survey collides on one cache entry.
                batch_key = _cache_key(
                    {**base_key, "batch": index,
                     "batch_qids": [q.question_id for q in batch]}
                )
                rp = build_question_prompt(
                    context, accumulator, taxonomy, questions=batch,
                )
                try:
                    result = loop.run_until_complete(
                        llm_client.complete(
                            rp.user_prompt, "",
                            cache_key=batch_key, call_type="question",
                            cached_system_preamble=rp.cached_preamble,
                            prompt_version=rp.version,
                        )
                    )
                except Exception as e:  # noqa: BLE001
                    # One bad batch must not cost the survey its other batches.
                    logger.warning("llm_question_batch_failed",
                                   extra={"batch": index, "of": len(batches),
                                          "error": str(e)})
                    continue

                if not (result and response_parser):
                    logger.warning("llm_question_batch_no_result",
                                   extra={"batch": index, "of": len(batches),
                                          "questions": len(batch)})
                    continue

                parsed_all.extend(response_parser.parse_question_response(
                    result, industry=industry, project_type=project_type,
                    journey=journey,
                ))
                calls_made += 1

            if parsed_all:
                _apply_question_llm_results(parsed_all, accumulator, context)
            # Runs once over the whole survey, after every batch has landed: a
            # question is only genuinely unplaced when no batch placed it.
            if calls_made:
                _close_unplaced_journeys(context, accumulator, journey)
                _close_unfilled_widget_fields(context, accumulator)
            logger.debug("llm_question_call_applied",
                         extra={"batches": len(batches),
                                "parsed_questions": len(parsed_all),
                                "journey_leaves": len(journey.leaves) if journey else 0})
    except Exception as e:
        logger.error(
            "llm_enhancement_failed type=%s: %s",
            type(e).__name__, e,
            extra={"error": str(e)},
            exc_info=True,
        )
    finally:
        loop.close()

    return calls_made


# ---------------------------------------------------------------------------
# Override policy: ONE place that answers "may the model replace this tag?"
#
# This was previously four different inline thresholds (0.80 for project scalars,
# 0.80 for question scalars, 0.80 for dashboard_placement, 0.70 for role_intent)
# plus a `status == "skipped"` check repeated in six places. Same decision, four
# spellings, no single place to read it — so the policy could not be audited and
# drifted every time a dimension was added.
# ---------------------------------------------------------------------------

# Default floor: a prior tag at or above this confidence keeps its value.
_DEFAULT_FLOOR = 0.80

# Per-dimension floors that differ from the default, each with its reason.
_FLOORS = {
    # role_intent's deterministic tiers are strong; only its weakest guesses
    # (< 0.70) are worth a model's second opinion.
    "role_intent": 0.70,
}

# Dimensions whose non-LLM tiers are authoritative even though the tagger reports
# `hybrid` rather than `deterministic`.
#
# `industry_vertical` is free text sourced from tenant research: when the org
# agent supplied a label the tagger stores it verbatim above the floor, and a
# model reading one survey does not get to rewrite what a researcher read off the
# tenant's own site. Every weaker tier of that tagger sits below the floor, so the
# threshold alone decides — no tier plumbing leaks in here.
_HYBRID_AUTHORITATIVE = frozenset({"industry_vertical"})


def _blocks_llm_override(existing, dimension: str, *, any_source: bool) -> bool:
    """True when the prior tag outranks whatever the model returned.

    Rules, in order:
      * a `skipped` tag is a decision, not a gap — the dimension does not apply
        here and the model does not get to apply it anyway;
      * below the dimension's floor, the model wins;
      * at or above it, `any_source` decides which sources are protected.

    `any_source` is the one place the two levels genuinely differ, and the
    difference is deliberate rather than drift:

      question level (`any_source=True`) — a question tagger that reached 0.80 read
        something structural about THAT question: a platform flag, its position, its
        grid size. `flow_experience` is the case that makes this matter: it reports
        `hybrid` at 0.80-0.85 for a welcome message or a section header, facts the
        model cannot see from the wording it is given.

      project level (`any_source=False`) — the free-text and refinement dimensions
        (project_intent, audience_type, survey_sub_type, ...) are seeded by rules
        from the title alone and the LLM, which reads every question, is the
        intended arbiter. Only deterministic tiers — and the hybrids named in
        `_HYBRID_AUTHORITATIVE` — outrank it.
    """
    if existing is None:
        return False
    if existing.status == "skipped":
        return True
    if existing.confidence < _FLOORS.get(dimension, _DEFAULT_FLOOR):
        return False
    if any_source or existing.source == "deterministic":
        return True
    return dimension in _HYBRID_AUTHORITATIVE and existing.source == "hybrid"


def _merge_multi_label(
    getter, setter, dimension: str, values, *, why: dict, why_key: str,
    summary: str | None, confidence: float, any_source: bool,
) -> None:
    """Union an LLM list onto an existing list tag, newest names appended.

    Union rather than replace: the deterministic pass and the model each see
    routing/placement signals the other cannot, and dropping either side loses
    information. A tag the policy protects is left alone entirely.

    Both list dimensions this serves (`dashboard_routing`, `dashboard_placement`)
    are emitted by their taggers at 0.60-0.70, i.e. always below the floor, so the
    policy check here never fires today. It is applied anyway so that raising a
    tagger's confidence cannot silently start overwriting it.
    """
    if not values or not isinstance(values, list):
        return
    existing = getter(dimension)
    if _blocks_llm_override(existing, dimension, any_source=any_source):
        return
    merged = list(existing.value) if existing and isinstance(existing.value, list) else []
    for name in values:
        if name not in merged:
            merged.append(name)
    setter(dimension, TagResult(
        value=merged, source="llm", confidence=confidence,
        reasoning=_rationale(why, why_key, summary),
        superseded=_superseded(existing),
    ))


def _apply_project_llm_results(parsed: dict, accumulator) -> None:
    """Apply LLM project-level results, upgrading low-confidence tags."""
    scalar_map = {
        "relationship_type": "relationship_type",
        "project_purpose": "project_purpose",
        # Free text, and the tagger's title-derived seed sits at 0.45 by design,
        # so the model's answer always wins here — the seed survives on
        # `superseded` rather than being erased.
        "project_intent": "project_intent",
        # Free text too, but the opposite rule: a profile-sourced industry is
        # authoritative (see `_HYBRID_AUTHORITATIVE`) and the model's answer is
        # only used when the tagger had nothing better than a seed.
        "industry_vertical": "industry_vertical",
        "audience_type_refined": "audience_type",
        "survey_sub_type": "survey_sub_type",
    }

    why = parsed.get("why") or {}
    summary = parsed.get("reasoning") or None
    get, set_ = accumulator.get_project_tag, accumulator.set_project_tag

    for field, dimension in scalar_map.items():
        # `field` is the model's key, `dimension` ours; the `why` line is filed
        # under the model's spelling.
        value = parsed.get(field)
        if not value:
            continue
        existing = get(dimension)
        if _blocks_llm_override(existing, dimension, any_source=False):
            continue
        set_(dimension, TagResult(
            value=value, source="llm", confidence=_PROJECT_CONFIDENCE,
            reasoning=_rationale(why, field, summary),
            superseded=_superseded(existing),
        ))

    # Multi-label: dashboard_names → dashboard_routing
    _merge_multi_label(get, set_, "dashboard_routing", parsed.get("dashboard_names"),
                       why=why, why_key="dashboard_names", summary=summary,
                       confidence=_PROJECT_CONFIDENCE, any_source=False)


def _close_unplaced_journeys(context, accumulator, journey) -> None:
    """Turn leftover `pending_llm` journey tags into skips that say why.

    `journey_stage` is a pending_llm-only tagger: the deterministic pass
    reserves the dimension and the LLM call fills it. Anything the call did not
    fill would otherwise keep a status whose documented meaning is "that call
    never landed" — which, once the call has landed and declined, is a lie the
    output tells about itself.

    V9 collapsed four reasons to two. `scoring_unavailable` (the embedding model
    would not load) and `below_similarity_floor` (nothing cleared `min_score`)
    both described the ranker, and the ranker is gone: every question with a
    journey now sees every moment in it. What is left is the honest pair —
      * no journey source for the tenant at all -> fetch the profile;
      * the whole journey was offered and the model declined -> nothing to fix.
    """
    from taggers._metric_utils import is_journey_eligible_metric

    for q in context.questions:
        if q.is_content_message:
            continue
        eligible, _ev = is_journey_eligible_metric(q)
        if not eligible:
            continue

        for dimension in ("journey_stage", "sub_stage_name"):
            existing = accumulator.get_question_tag(q.question_id, dimension)
            if existing is None or existing.status != "pending_llm":
                continue

            if journey is None:
                detail = ev.fallback(
                    f"question.{dimension}.no_journey_source",
                    "No journey is available for this tenant: `tenant_profile/` carries "
                    "no CX journeys or EX lifecycle stages to place this metric against. "
                    "Fetch the tenant profile and re-tag to populate this dimension.",
                    stage=5,
                    inputs={"tenant_id": context.tenant_id},
                )
            else:
                detail = ev.rule(
                    f"question.{dimension}.llm_declined",
                    f"All {len(journey.leaves)} moments in '{journey.journey_name}' were "
                    "offered and the model judged that none of them is what this "
                    "question measures.",
                    stage=5,
                    inputs={"leaves_offered": len(journey.leaves)},
                )

            accumulator.set_question_tag(q.question_id, dimension, TagResult(
                value=None, source="deterministic", status="skipped", evidence=detail,
            ))


# The V8 widget fields a tagger may leave `pending_llm`, with what an unfilled
# one means once the call HAS landed and declined. Each sentence is the honest
# reading of that decline, not an error — the prompt tells the model to omit
# these fields when they do not apply, so most of these are correct answers.
_WIDGET_FIELD_DECLINED: dict[str, str] = {
    "favorable_options": (
        "No favorability split was recorded: either the question's role marks its "
        "options as alternatives (a demographic, a segmentation list, a screener), so "
        "it was never asked about, or the options were shown to the model with their "
        "ids and it returned nothing. Both are the right answer for a list that is not "
        "a good-to-bad scale — Percent Favorable and Net Intent are not metrics this "
        "question can carry."
    ),
    "preferred_segments": (
        "The survey's segmentable questions were offered as candidates and the model "
        "judged that none of them is a cut of this metric worth showing. An unsegmented "
        "widget is a correct outcome, not a gap."
    ),
    "display_label": (
        "The question has no metric name and no usable title text, and the model wrote "
        "no label from its options or section header. The widget will fall back to the "
        "question text."
    ),
}


def _close_unfilled_widget_fields(context, accumulator) -> None:
    """Turn leftover `pending_llm` widget tags into skips that say why.

    Same reasoning as `_close_unplaced_journeys`: `pending_llm` documents "that
    call never landed", so once it has landed and declined, leaving the status
    in place is a lie the output tells about itself. Only runs when at least one
    question call succeeded — a survey whose batches all failed keeps
    `pending_llm`, which is then true.
    """
    for q in context.questions:
        if q.is_content_message:
            continue
        for dimension, reason in _WIDGET_FIELD_DECLINED.items():
            existing = accumulator.get_question_tag(q.question_id, dimension)
            if existing is None or existing.status != "pending_llm":
                continue
            accumulator.set_question_tag(q.question_id, dimension, TagResult(
                value=[] if dimension == "preferred_segments" else None,
                source="deterministic", status="skipped",
                evidence=ev.rule(
                    f"question.{dimension}.llm_declined", reason, stage=5,
                    inputs={"question_type": q.question_type},
                ),
            ))


# Confidence for a role the model read out of question wording. Deliberately
# well under the deterministic pass's 0.90: those roles are read off platform
# flags, these are an inference about branching nobody configured in the payload.
_FLOW_LOGIC_INFERRED_CONFIDENCE = 0.55


def _apply_flow_logic_inference(
    q_data: dict, accumulator, q_id: int, why: dict, q_reasoning: str | None
) -> None:
    """Union the model's inferred routing roles onto the deterministic ones.

    Three rules, and the first is the one that matters: the LLM may only ADD.
    The deterministic entries are read off platform flags (`questionType == HR`,
    `isFollowupQuestion`, `metricQuestion`, piping markers) — facts — while these
    are inferred from how the question is worded. A model that could delete a
    fact would be trading a certainty for a guess.

    Second: a question that already carries the inferred role keeps its
    deterministic entry, so the union never demotes a 0.90 fact to a 0.55 guess.
    Third: the resulting tag is `hybrid` only when both passes contributed —
    `source` has to keep answering "where did this come from?" honestly, and a
    tag holding nothing but deterministic roles is not a hybrid of anything.
    """
    inferred = q_data.get("flow_logic_inferred")
    if not inferred or not isinstance(inferred, list):
        return

    existing = accumulator.get_question_tag(q_id, "flow_logic_role")
    if existing is not None and existing.status == "skipped":
        return  # content message — it has no flow role to reason about

    prior = list(existing.value) if existing and isinstance(existing.value, list) else []
    added = [r for r in inferred if r not in prior]
    if not added:
        return

    merged = prior + added
    detail = (
        f"{len(added)} routing role(s) inferred from the question's wording and "
        f"position: {', '.join(added)}. The survey payload carries no skip-logic or "
        "branching definitions at all, so nothing here was read off a configured "
        "rule — this says the question READS like a trigger, which is a lead to "
        "verify, not a statement that branching exists."
    )
    if prior:
        detail += (
            f" Merged onto {len(prior)} role(s) the structural pass already "
            f"established ({', '.join(prior)}); those are platform facts and were "
            "kept as they were."
        )

    accumulator.set_question_tag(q_id, "flow_logic_role", TagResult(
        value=merged,
        source="hybrid" if prior else "llm",
        confidence=_FLOW_LOGIC_INFERRED_CONFIDENCE,
        evidence=ev.hybrid(
            "question.flow_logic_role.llm_inferred",
            detail,
            components=[ev.component(r, "inferred from question wording") for r in added]
            + [ev.component(r, "detected structurally") for r in prior],
            stage=5,
            inputs={"inferred": added, "structural": prior},
        ),
        reasoning=_rationale(why, "flow_logic_inferred", q_reasoning),
        # Only when a rule actually said something. `_superseded` guards on
        # `value is None`, and the no-logic tag's value is `[]` — recording
        # "the rule previously said nothing" is noise, not provenance.
        superseded=_superseded(existing) if prior else None,
    ))


# Confidence for the three V8 widget fields. The same 0.80 every other
# LLM-assigned question tag carries: high enough to beat the seeds the taggers
# deliberately hold below it, low enough that a future deterministic rule wins.
_WIDGET_FIELD_CONFIDENCE = 0.80


def _apply_widget_fields(
    q_data: dict, accumulator, q_id: int, question, why: dict, q_reasoning: str | None,
) -> None:
    """Apply favorable_options, preferred_segments and display_label.

    This is where MEMBERSHIP is enforced, because this is the first place that
    holds the question and the survey. The parser guaranteed clean int lists;
    what it could not know is whether those ids exist. Both id fields are
    filtered against the namespace the prompt actually offered, so a model that
    invents an option or names a question that cannot segment writes nothing
    rather than writing a dangling reference into the payload.

    Each field is written only over a tag the tagger left open (`pending_llm`,
    or a seed below the override threshold) — the same merge rule the scalar
    fields use, applied through `_blocks_llm_override`'s sibling test here so a
    settled rule always wins.
    """
    # --- favorable_options: ids must be this question's own answer options ---
    favorable = q_data.get("favorable_options")
    if favorable and question is not None:
        valid = {o.answer_id for o in question.answer_options}
        filtered = {b: [i for i in ids if i in valid] for b, ids in favorable.items()}
        dropped = sum(len(ids) for ids in favorable.values()) - sum(
            len(ids) for ids in filtered.values())
        if any(filtered.values()):
            existing = accumulator.get_question_tag(q_id, "favorable_options")
            if not (existing and (existing.status == "skipped"
                                  or existing.confidence >= _WIDGET_FIELD_CONFIDENCE)):
                line = _rationale(why, "favorable_options", q_reasoning)
                if dropped:
                    note = (f"{dropped} answer id(s) the model returned are not options "
                            "on this question and were dropped.")
                    line = f"{line} — {note}" if line else note
                accumulator.set_question_tag(q_id, "favorable_options", TagResult(
                    value=filtered, source="llm",
                    confidence=_WIDGET_FIELD_CONFIDENCE,
                    reasoning=line, superseded=_superseded(existing),
                ))

    # --- preferred_segments: ids must be segmentable questions in this survey ---
    segments = q_data.get("preferred_segments")
    if segments:
        filtered = [
            qid for qid in segments
            if qid != q_id
            and accumulator.get_question_tag_value(qid, "is_segmentable") == "Yes"
        ]
        if filtered:
            existing = accumulator.get_question_tag(q_id, "preferred_segments")
            if not (existing and (existing.status == "skipped"
                                  or existing.confidence >= _WIDGET_FIELD_CONFIDENCE)):
                line = _rationale(why, "preferred_segments", q_reasoning)
                if len(filtered) != len(segments):
                    note = (f"{len(segments) - len(filtered)} id(s) the model returned "
                            "are not segmentable questions in this survey and were "
                            "dropped.")
                    line = f"{line} — {note}" if line else note
                accumulator.set_question_tag(q_id, "preferred_segments", TagResult(
                    value=filtered, source="llm",
                    confidence=_WIDGET_FIELD_CONFIDENCE,
                    reasoning=line, superseded=_superseded(existing),
                ))

    # --- display_label: free text, nothing to check membership against ---
    label = q_data.get("display_label")
    if label:
        existing = accumulator.get_question_tag(q_id, "display_label")
        if not (existing and (existing.status == "skipped"
                              or existing.confidence >= _WIDGET_FIELD_CONFIDENCE)):
            accumulator.set_question_tag(q_id, "display_label", TagResult(
                value=label, source="llm", confidence=_WIDGET_FIELD_CONFIDENCE,
                reasoning=_rationale(why, "display_label", q_reasoning),
                superseded=_superseded(existing),
            ))


def _apply_question_llm_results(parsed_list: list[dict], accumulator, context=None) -> None:
    """Apply LLM question-level results, including v2 + v5 fields.

    For journey_stage and sub_stage_name, enforce metric eligibility regardless
    of what the LLM returned.
    """
    from taggers._metric_utils import is_journey_eligible_metric

    question_by_id = (
        {q.question_id: q for q in context.questions} if context is not None else {}
    )

    for q_data in parsed_list:
        q_id = q_data.get("id")
        if not q_id:
            continue

        scalar_map = {
            "topic_theme": "topic_theme",
            "respondent_sensitivity": "respondent_sensitivity",
            "flow_respondent_experience": "flow_respondent_experience",
            "flow_reusability": "flow_reusability",
            "visualization_type": "visualization_type",
            "display_role": "display_role",
        }

        question = question_by_id.get(q_id)

        # V7.1: one rationale line per dimension (`why`), with the question-level
        # summary (`reasoning`) as the fallback for any dimension the model left
        # unexplained. Each LLM-sourced tag then explains its own value rather
        # than repeating a blob that covers seven other dimensions.
        why = q_data.get("why") or {}
        q_reasoning = q_data.get("reasoning") or None

        # Both helpers below take (dimension, ...) — bind the question id once.
        def get(dim, _q=q_id):
            return accumulator.get_question_tag(_q, dim)

        def set_(dim, tag, _q=q_id):
            accumulator.set_question_tag(_q, dim, tag)

        # -------- Standard scalar fields --------
        for field, dimension in scalar_map.items():
            value = q_data.get(field)
            if not value:
                continue
            existing = get(dimension)
            if _blocks_llm_override(existing, dimension, any_source=True):
                continue
            set_(dimension, TagResult(
                value=value, source="llm", confidence=_QUESTION_CONFIDENCE,
                reasoning=_rationale(why, field, q_reasoning),
                superseded=_superseded(existing),
            ))

        # -------- Journey block (atomic stage + sub_stage) --------
        stage_value = q_data.get("journey_stage")
        sub_value = q_data.get("sub_stage_name")
        j_status = q_data.get("journey_status") or "assigned"
        j_confidence = q_data.get("journey_confidence") or "medium"
        j_evidence = q_data.get("journey_evidence")

        if stage_value or sub_value:
            if question is not None:
                eligible, _ev = is_journey_eligible_metric(question)
                if not eligible:
                    continue

            conf_numeric = _CONF_NUMERIC.get(j_confidence, 0.75)
            tag_status = "low_confidence_assigned" if j_status == "low_confidence_assigned" else "assigned"

            # V9: `candidates` is gone. It held this question's ranked top-4,
            # and there is no longer a per-question shortlist — the model chose
            # from the tenant's whole journey, which is recorded by name and
            # size instead of copied into every question's tag.
            coverage = {
                "confidence": j_confidence,
                "evidence": j_evidence,
                "leaf_id": q_data.get("journey_leaf_id"),
                "journey_name": q_data.get("journey_name"),
                "leaves_offered": q_data.get("journey_leaves_offered"),
            }

            # The journey block carries its own per-assignment sentence
            # (`journey.evidence`), so that is the rationale here — the generic
            # question summary is only the last resort.
            j_reasoning = j_evidence or q_reasoning

            if stage_value:
                existing = accumulator.get_question_tag(q_id, "journey_stage")
                if not (existing and existing.status == "skipped"):
                    accumulator.set_question_tag(q_id, "journey_stage", TagResult(
                        value=stage_value, source="llm",
                        confidence=conf_numeric, status=tag_status,
                        evidence=j_evidence, reasoning=j_reasoning,
                        coverage_metadata=coverage,
                    ))
            existing_sub = accumulator.get_question_tag(q_id, "sub_stage_name")
            if not (existing_sub and existing_sub.status == "skipped"):
                if sub_value:
                    accumulator.set_question_tag(q_id, "sub_stage_name", TagResult(
                        value=sub_value, source="llm",
                        confidence=conf_numeric, status=tag_status,
                        evidence=j_evidence, reasoning=j_reasoning,
                        coverage_metadata=coverage,
                    ))
                elif stage_value:
                    # A one-level journey (the EX lifecycle) has no sub-stage to
                    # assign. Record that as a skip against the source, not as a
                    # question we failed to reach — and never synthesize a label,
                    # which is how metric names ended up in this column.
                    accumulator.set_question_tag(q_id, "sub_stage_name", TagResult(
                        value=None, source="deterministic", status="skipped",
                        evidence=ev.rule(
                            "question.sub_stage_name.source_has_one_level",
                            f"Placed at '{stage_value}', but the tenant's journey for this "
                            "survey type is a flat stage list with no sub-stage level, so "
                            "there is nothing to assign.",
                            stage=5,
                        ),
                    ))

        # Multi-label: dashboard_names → dashboard_placement
        _merge_multi_label(get, set_, "dashboard_placement",
                           q_data.get("dashboard_names"),
                           why=why, why_key="dashboard_names", summary=q_reasoning,
                           confidence=_QUESTION_CONFIDENCE, any_source=True)

        # Multi-label: flow_logic_inferred → flow_logic_role (union, never removal)
        _apply_flow_logic_inference(q_data, accumulator, q_id, why, q_reasoning)

        # V8 widget fields, with their id namespaces enforced.
        _apply_widget_fields(q_data, accumulator, q_id, question, why, q_reasoning)

        # Role intent refinement. `_FLOORS` holds role_intent's 0.70 floor, and
        # the `why` line arrives under the model's own key.
        refined_role = q_data.get("role_intent_refined")
        if refined_role:
            existing_role = get("role_intent")
            if not _blocks_llm_override(existing_role, "role_intent", any_source=True):
                set_("role_intent", TagResult(
                    value=refined_role,
                    source="llm",
                    confidence=_QUESTION_CONFIDENCE,
                    reasoning=_rationale(why, "role_intent_refined", q_reasoning),
                    superseded=_superseded(existing_role),
                ))


