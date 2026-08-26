"""The per-survey tagging engine.

`process_single_survey` is the single unit of survey tagging. It runs the
deterministic/statistical/hybrid tagger stages and, when an LLM client is
supplied, the Stage 4–5 LLM enhancement — then assembles the output via the
shared assembler. The tenant orchestrator calls this once per survey, so a
survey tagged on its own and the same survey tagged in a tenant run produce
identical output.

Without `llm_client` it is deterministic-only (no LLM, no network). It performs
no filesystem I/O and no change detection — those are the orchestrator's job.
"""

from __future__ import annotations

import logging
import time

from models.context import UnifiedContext
from models.tags import TagAccumulator, TaggedSurvey
from models.taxonomy import TaxonomyRegistry
from pipeline.assembly import assemble_tagged_survey
from taggers.base import ProjectTagger, QuestionTagger
from taggers.registry import TaggerRegistry

logger = logging.getLogger(__name__)


def process_single_survey(
    context: UnifiedContext,
    registry: TaggerRegistry,
    taxonomy: TaxonomyRegistry,
    *,
    llm_client=None,
    response_parser=None,
    settings=None,
) -> TaggedSurvey:
    """Run all taggers on a single survey context.

    Args:
        context: Pre-assembled UnifiedContext.
        registry: Tagger registry with all taggers discovered.
        taxonomy: Taxonomy registry for validation.
        llm_client: Optional LLMClient. When provided (and not skip_llm), the
            Stage 4–5 LLM enhancement runs; otherwise tagging is deterministic.
        response_parser: ResponseParser required for LLM enhancement.
        settings: Settings (for skip_llm + embedding config). Optional.

    Returns:
        TaggedSurvey with all tags assigned.
    """
    start_time = time.time()
    accumulator = TagAccumulator()

    stages = registry.resolve_execution_order()
    logger.debug(
        "single_survey_start",
        extra={
            "tenant": context.tenant_id,
            "survey": context.survey_meta.survey_no,
            "survey_name": context.survey_meta.title,
            "questions": len(context.questions),
            "stage_count": len(stages),
        },
    )

    for stage_idx, stage_group in enumerate(stages, start=1):
        logger.debug(
            "stage_start",
            extra={"stage_index": stage_idx, "taggers": [t.name for t in stage_group]},
        )
        for tagger in stage_group:
            try:
                if isinstance(tagger, ProjectTagger):
                    result = tagger.tag(context, accumulator)
                    accumulator.set_project_tag(tagger.tag_dimension, result)
                    logger.debug(
                        "project_tag_assigned",
                        extra={"tagger": tagger.name, "dimension": tagger.tag_dimension,
                               "value": result.value, "source": result.source,
                               "confidence": result.confidence, "status": result.status},
                    )
                elif isinstance(tagger, QuestionTagger):
                    for question in context.questions:
                        result = tagger.tag_question(context, question, accumulator)
                        accumulator.set_question_tag(
                            question.question_id, tagger.tag_dimension, result
                        )
                    logger.debug(
                        "question_tag_assigned",
                        extra={"tagger": tagger.name, "dimension": tagger.tag_dimension,
                               "questions_tagged": len(context.questions)},
                    )
            except Exception as e:
                logger.warning("tagger_failed", extra={"tagger": tagger.name, "error": str(e)})
                accumulator.mark_failed(tagger.tag_dimension, str(e))

    # LLM enhancement (Stages 4–5), if a client is available and not skipped.
    llm_calls = 0
    skip_llm = bool(settings.skip_llm) if settings is not None else False
    if llm_client is not None and response_parser is not None and not skip_llm:
        from pipeline.llm_enhance import run_llm_enhancement
        logger.debug("llm_enhancement_start", extra={"survey": context.survey_meta.survey_no})
        llm_calls = run_llm_enhancement(
            context, accumulator,
            llm_client=llm_client, taxonomy=taxonomy,
            response_parser=response_parser, settings=settings,
        )
        logger.debug("llm_enhancement_done",
                     extra={"survey": context.survey_meta.survey_no, "llm_calls_made": llm_calls})

    elapsed = int((time.time() - start_time) * 1000)
    logger.debug(
        "single_survey_tagging_complete",
        extra={"survey": context.survey_meta.survey_no,
               "project_dimensions": len(accumulator.project_tags),
               "llm_calls_made": llm_calls, "elapsed_ms": elapsed},
    )

    return assemble_tagged_survey(context, accumulator, llm_calls, elapsed)
