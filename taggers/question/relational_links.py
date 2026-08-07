"""Cross-question relationship taggers (V7) — surface raw platform linkage
signals already loaded onto QuestionContext so the downstream dashboard service
can PAIR questions into combined widgets.

Three Stage-3 deterministic, user_defined dimensions:

    driver_link ..... is_key_driver           -> pair driver to its outcome metric
    verbatim_link ... metric_question_id       -> attach follow-up/verbatim to parent
    block_id ........ matrix_group_title/size  -> group matrix rows into one widget

Each returns status="skipped" when the relationship does not apply. Values are
references (ids / group names), hence user_defined (no allowed_values check).
"""

from __future__ import annotations

from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger


class DriverLinkTagger(QuestionTagger):
    name = "question.driver_link"
    tag_dimension = "driver_link"
    stage = 3
    source_type = "deterministic"

    def tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        q = question
        if q.is_content_message:
            return TagResult(value=None, source="deterministic", status="skipped",
                             evidence="Content message")
        if not q.is_key_driver:
            return TagResult(value=None, source="deterministic", status="skipped",
                             evidence="Not a key driver")
        # Reference the outcome metric when known, else flag as a key driver.
        if q.metric_question_id:
            return TagResult(value=str(q.metric_question_id), source="deterministic",
                             confidence=1.0, evidence="is_key_driver -> outcome metric_question_id")
        return TagResult(value="Key Driver", source="deterministic", confidence=1.0,
                         evidence="is_key_driver=True (outcome metric unresolved)")


class VerbatimLinkTagger(QuestionTagger):
    name = "question.verbatim_link"
    tag_dimension = "verbatim_link"
    stage = 3
    source_type = "deterministic"

    def tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        q = question
        if q.is_content_message:
            return TagResult(value=None, source="deterministic", status="skipped",
                             evidence="Content message")
        if q.metric_question_id:
            return TagResult(value=str(q.metric_question_id), source="deterministic",
                             confidence=1.0,
                             evidence=f"Follow-up of parent metric {q.metric_question_id}")
        return TagResult(value=None, source="deterministic", status="skipped",
                         evidence="Standalone (no parent metric)")


class BlockIdTagger(QuestionTagger):
    name = "question.block_id"
    tag_dimension = "block_id"
    stage = 3
    source_type = "deterministic"

    def tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        q = question
        if q.is_content_message:
            return TagResult(value=None, source="deterministic", status="skipped",
                             evidence="Content message")
        if q.matrix_group_size and q.matrix_group_size > 1 and q.matrix_group_title:
            return TagResult(value=q.matrix_group_title, source="deterministic",
                             confidence=1.0,
                             evidence=f"Matrix block '{q.matrix_group_title}' "
                                      f"({q.matrix_group_size} rows)")
        return TagResult(value=None, source="deterministic", status="skipped",
                         evidence="Standalone (not part of a matrix block)")


def create_tagger() -> list[QuestionTagger]:
    return [DriverLinkTagger(), VerbatimLinkTagger(), BlockIdTagger()]
