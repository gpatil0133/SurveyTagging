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
from models.tags import TagResult

logger = logging.getLogger(__name__)

# LLM confidence string → numeric for accumulator merge ranking.
_CONF_NUMERIC = {"high": 0.85, "medium": 0.75, "low": 0.55, "none": 0.40}


def run_llm_enhancement(
    context,
    accumulator,
    *,
    llm_client,
    taxonomy,
    industry_stages,
    response_parser,
    settings,
) -> int:
    """Run LLM calls for semantic tagging. Returns number of calls made."""
    calls_made = 0

    survey_hash = LLMCache.compute_hash(
        json.dumps({"title": context.survey_meta.title,
                    "questions": [q.title for q in context.questions]})
    )
    logger.debug("llm_cache_key_computed", extra={"survey_hash": survey_hash})

    try:
        loop = asyncio.new_event_loop()

        # LLM Call 1: Project-level
        logger.debug("llm_project_call_start", extra={"survey_hash": survey_hash})
        rp = build_project_prompt(context, accumulator, taxonomy)
        logger.debug("project_prompt_built",
                     extra={"prompt_version": rp.version,
                            "user_prompt_chars": len(rp.user_prompt),
                            "cached_preamble_chars": len(rp.cached_preamble or "")})
        result = loop.run_until_complete(
            llm_client.complete(
                rp.user_prompt, "",
                cache_key=survey_hash, call_type="project",
                cached_system_preamble=rp.cached_preamble,
                prompt_version=rp.version,
            )
        )
        if result and response_parser:
            parsed = response_parser.parse_project_response(result)
            _apply_project_llm_results(parsed, accumulator)
            calls_made += 1
            logger.debug("llm_project_call_applied", extra={"survey_hash": survey_hash})
        else:
            logger.debug("llm_project_call_no_result",
                         extra={"survey_hash": survey_hash,
                                "has_result": result is not None})

        # LLM Call 2: Question-level
        if context.non_cm_questions and industry_stages:
            logger.debug("llm_question_call_start",
                         extra={"survey_hash": survey_hash,
                                "non_cm_questions": len(context.non_cm_questions)})
            rp = build_question_prompt(context, accumulator, taxonomy, industry_stages)
            result = loop.run_until_complete(
                llm_client.complete(
                    rp.user_prompt, "",
                    cache_key=survey_hash, call_type="question",
                    cached_system_preamble=rp.cached_preamble,
                    prompt_version=rp.version,
                )
            )
            if result and response_parser:
                industry = accumulator.get_project_tag_value("industry_vertical")
                project_type = accumulator.get_project_tag_value("project_type")
                # Select CX vs EX canon by project_type (EX surveys ground
                # against the employee-lifecycle canon).
                canon, canon_embeddings = context.canon_for(project_type)
                # Capture per-question candidates so the parser can downgrade
                # out-of-canon stage names to top-1 instead of dropping them.
                candidates_by_qid = _gather_question_candidates(context, settings, canon_embeddings)
                parsed_questions = response_parser.parse_question_response(
                    result, industry=industry, project_type=project_type,
                    canon=canon,
                    candidates_by_qid=candidates_by_qid,
                )
                _apply_question_llm_results(parsed_questions, accumulator, context)
                calls_made += 1
                logger.debug("llm_question_call_applied",
                             extra={"survey_hash": survey_hash,
                                    "parsed_questions": len(parsed_questions),
                                    "candidates_qids": len(candidates_by_qid)})

        loop.close()
    except Exception as e:
        logger.error(
            "llm_enhancement_failed type=%s: %s",
            type(e).__name__, e,
            extra={"error": str(e)},
            exc_info=True,
        )

    return calls_made


def _apply_project_llm_results(parsed: dict, accumulator) -> None:
    """Apply LLM project-level results, upgrading low-confidence tags."""
    scalar_map = {
        "relationship_type": "relationship_type",
        "project_purpose": "project_purpose",
        "industry_vertical": "industry_vertical",
        "audience_type_refined": "audience_type",
        "survey_sub_type": "survey_sub_type",
    }

    for field, dimension in scalar_map.items():
        value = parsed.get(field)
        if not value:
            continue

        existing = accumulator.get_project_tag(dimension)
        if existing and existing.confidence >= 0.80 and existing.source == "deterministic":
            continue

        accumulator.set_project_tag(dimension, TagResult(
            value=value,
            source="llm",
            confidence=0.85,
            reasoning=parsed.get("reasoning", ""),
        ))

    # Multi-label: dashboard_names → dashboard_routing
    dashboard_names = parsed.get("dashboard_names")
    if dashboard_names and isinstance(dashboard_names, list):
        existing = accumulator.get_project_tag("dashboard_routing")
        if existing and isinstance(existing.value, list):
            merged = list(existing.value)
            for n in dashboard_names:
                if n not in merged:
                    merged.append(n)
            value = merged
        else:
            value = dashboard_names
        accumulator.set_project_tag("dashboard_routing", TagResult(
            value=value, source="llm", confidence=0.85,
            reasoning=parsed.get("reasoning", ""),
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

        # -------- Standard scalar fields --------
        for field, dimension in scalar_map.items():
            value = q_data.get(field)
            if not value:
                continue
            existing = accumulator.get_question_tag(q_id, dimension)
            if existing and (existing.status == "skipped" or existing.confidence >= 0.80):
                continue
            accumulator.set_question_tag(q_id, dimension, TagResult(
                value=value, source="llm", confidence=0.80,
            ))

        # -------- Journey block (atomic stage + sub_stage) --------
        stage_value = q_data.get("journey_stage")
        sub_value = q_data.get("sub_stage_name")
        j_status = q_data.get("journey_status") or "assigned"
        j_confidence = q_data.get("journey_confidence") or "medium"
        j_evidence = q_data.get("journey_evidence")
        j_candidates = q_data.get("journey_candidates")

        if stage_value or sub_value:
            if question is not None:
                eligible, _ev = is_journey_eligible_metric(question)
                if not eligible:
                    continue

            conf_numeric = _CONF_NUMERIC.get(j_confidence, 0.75)
            tag_status = "low_confidence_assigned" if j_status == "low_confidence_assigned" else "assigned"

            coverage = {
                "confidence": j_confidence,
                "evidence": j_evidence,
                "candidates": j_candidates or [],
            }

            if stage_value:
                existing = accumulator.get_question_tag(q_id, "journey_stage")
                if not (existing and existing.status == "skipped"):
                    accumulator.set_question_tag(q_id, "journey_stage", TagResult(
                        value=stage_value, source="llm",
                        confidence=conf_numeric, status=tag_status,
                        evidence=j_evidence, coverage_metadata=coverage,
                    ))
            if sub_value:
                existing_sub = accumulator.get_question_tag(q_id, "sub_stage_name")
                if not (existing_sub and existing_sub.status == "skipped"):
                    accumulator.set_question_tag(q_id, "sub_stage_name", TagResult(
                        value=sub_value, source="llm",
                        confidence=conf_numeric, status=tag_status,
                        evidence=j_evidence, coverage_metadata=coverage,
                    ))

        # Multi-label: dashboard_names → dashboard_placement
        dashboard_names = q_data.get("dashboard_names")
        if dashboard_names and isinstance(dashboard_names, list):
            existing = accumulator.get_question_tag(q_id, "dashboard_placement")
            if existing and isinstance(existing.value, list) and existing.confidence < 0.80:
                merged = list(existing.value)
                for n in dashboard_names:
                    if n not in merged:
                        merged.append(n)
                value = merged
            else:
                value = dashboard_names
            accumulator.set_question_tag(q_id, "dashboard_placement", TagResult(
                value=value, source="llm", confidence=0.80,
            ))

        # Role intent refinement
        refined_role = q_data.get("role_intent_refined")
        if refined_role:
            existing_role = accumulator.get_question_tag(q_id, "role_intent")
            if existing_role and existing_role.confidence < 0.70:
                accumulator.set_question_tag(q_id, "role_intent", TagResult(
                    value=refined_role,
                    source="llm",
                    confidence=0.80,
                ))


def _gather_question_candidates(context, settings, canon_embeddings=None) -> dict[int, list[dict]]:
    """Re-run per-question canon scoring to produce candidates_by_qid for the
    response parser's out-of-canon top-1 fallback. Returns {} when embeddings
    are unavailable.

    `canon_embeddings` is the journey-type-selected index (CX or EX) for this
    survey, resolved by the caller via `context.canon_for(project_type)`.
    """
    if canon_embeddings is None:
        return {}
    try:
        from llm.embeddings import EmbeddingModel, score_signature
        from llm.prompt_builder import build_question_signature
        from taggers._metric_utils import is_journey_eligible_metric

        embedder = EmbeddingModel.get(settings.embedding_model)
        top_k = settings.embedding_top_k

        out: dict[int, list[dict]] = {}
        for q in context.questions:
            if q.is_content_message:
                continue
            eligible, _ev = is_journey_eligible_metric(q)
            if not eligible:
                continue
            sig = build_question_signature(context, q)
            ranked = score_signature(sig, canon_embeddings, embedder, top_k=top_k)
            out[q.question_id] = [
                {"stage_name": s.name, "score": round(score, 3)}
                for (s, score) in ranked
            ]
        return out
    except Exception as e:  # noqa: BLE001
        logger.warning("question_candidate_gather_failed", extra={"error": str(e)})
        return {}
