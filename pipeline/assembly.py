"""Single output assembler: TagAccumulator + context -> TaggedSurvey.

One place builds the tagged-survey payload. Both the per-survey engine
(`pipeline.single_survey`) and the tenant orchestrator delegate here, so a
survey tagged on its own and the same survey tagged in a tenant run produce
identical output.
"""

from __future__ import annotations

from models.tags import TagAccumulator, TaggedQuestion, TaggedSurvey


def assemble_tagged_survey(
    context,
    accumulator: TagAccumulator,
    llm_calls: int,
    elapsed_ms: int,
) -> TaggedSurvey:
    """Pack accumulated tags into the final TaggedSurvey."""
    # ----- Project tags -----
    project_tags: dict[str, dict] = {}
    for dim, tag in accumulator.project_tags.items():
        entry: dict = {
            "value": tag.value,
            "source": tag.source,
            "confidence": tag.confidence,
        }
        if tag.evidence:
            entry["evidence"] = tag.evidence
        if tag.reasoning:
            entry["reasoning"] = tag.reasoning
        if tag.apply_method != "System-applied":
            entry["apply_method"] = tag.apply_method
        project_tags[dim] = entry

    # ----- Question tags -----
    question_tags: list[TaggedQuestion] = []
    for q in context.questions:
        q_tags_raw = accumulator.question_tags.get(q.question_id, {})
        q_tags: dict[str, dict] = {}
        for dim, tag in q_tags_raw.items():
            if tag.status == "skipped":
                continue
            t_entry: dict = {"value": tag.value, "source": tag.source}
            if tag.confidence < 1.0:
                t_entry["confidence"] = tag.confidence
            if tag.evidence:
                t_entry["evidence"] = tag.evidence
            # Preserve LLM-grounded candidates/confidence on journey_stage /
            # sub_stage_name so the survey-view endpoint can expose "why this
            # stage". No-op for tags that don't carry coverage_metadata.
            if tag.coverage_metadata:
                t_entry["coverage_metadata"] = tag.coverage_metadata
            q_tags[dim] = t_entry

        question_tags.append(TaggedQuestion(
            question_id=q.question_id,
            question_no=q.question_no,
            question_title_preview=q.title[:100],
            question_text=q.title,
            is_content_message=q.is_content_message,
            rs_type=q.rs_type,
            is_custom_metric=q.is_custom_metric,
            tags=q_tags,
        ))

    # ----- Context sources actually used -----
    sources_used = ["survey_structure"]
    if not context.overrides.is_empty:
        sources_used.append("manual_overrides")
    if context.tenant_profile is not None and not context.tenant_profile.is_empty:
        sources_used.append("tenant_profile")
    if context.has_responses:
        sources_used.append("response_data")
    if not context.directory_signals.is_empty:
        sources_used.append("directory_data")
    if context.invitation_signals:
        sources_used.append("invitation_data")

    low_conf = [
        dim for dim, tag in accumulator.project_tags.items()
        if tag.confidence < 0.60
    ]

    return TaggedSurvey(
        tenant_id=context.tenant_id,
        survey_no=context.survey_meta.survey_no,
        zarca_id=context.survey_meta.zarca_id,
        survey_name=context.survey_meta.title,
        project_tags=project_tags,
        question_tags=question_tags,
        metadata={
            "context_sources_used": sources_used,
            "llm_calls_made": llm_calls,
            "total_questions": len(context.questions),
            "questions_tagged": len([q for q in question_tags if not q.is_content_message]),
            "questions_skipped": len([q for q in question_tags if q.is_content_message]),
            "low_confidence_flags": low_conf,
            "processing_time_ms": elapsed_ms,
        },
    )
