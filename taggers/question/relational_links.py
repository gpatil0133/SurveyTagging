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

from models import evidence as ev
from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger


class DriverLinkTagger(QuestionTagger):
    name = "question.driver_link"
    tag_dimension = "driver_link"
    stage = 3
    source_type = "deterministic"

    def _tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        q = question
        if not q.is_key_driver:
            return TagResult(
                value=None, source="deterministic", status="skipped",
                evidence=ev.rule(
                    "question.driver_link.not_a_driver",
                    "The platform does not flag this question as a key driver, so "
                    "there is no outcome metric to pair it with in a combined widget.",
                    stage=3,
                    inputs={"is_key_driver": False},
                ),
            )
        # Reference the outcome metric when known, else flag as a key driver.
        if q.metric_question_id:
            return TagResult(
                value=str(q.metric_question_id), source="deterministic", confidence=1.0,
                evidence=ev.rule(
                    "question.driver_link.resolved_outcome",
                    f"The platform flags this as a key driver and names question "
                    f"{q.metric_question_id} as the outcome it drives, so the two can "
                    "be rendered as one driver-vs-outcome widget.",
                    stage=3,
                    inputs={"is_key_driver": True,
                            "outcome_question_id": q.metric_question_id},
                ),
            )
        return TagResult(
            value="Key Driver", source="deterministic", confidence=1.0,
            evidence=ev.rule(
                "question.driver_link.unresolved_outcome",
                "The platform flags this as a key driver but names no outcome "
                "question, so the link cannot point anywhere. The literal value "
                '"Key Driver" marks it as a driver without a partner to pair with.',
                stage=3,
                inputs={"is_key_driver": True, "outcome_question_id": None},
            ),
        )


class VerbatimLinkTagger(QuestionTagger):
    name = "question.verbatim_link"
    tag_dimension = "verbatim_link"
    stage = 3
    source_type = "deterministic"

    def _tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        q = question
        if q.metric_question_id:
            return TagResult(
                value=str(q.metric_question_id), source="deterministic", confidence=1.0,
                evidence=ev.rule(
                    "question.verbatim_link.has_parent",
                    f"This question follows on from question {q.metric_question_id}, "
                    "so its answers explain that metric's score and should be read "
                    "attached to it rather than on their own.",
                    stage=3,
                    inputs={"parent_question_id": q.metric_question_id},
                ),
            )
        return TagResult(
            value=None, source="deterministic", status="skipped",
            evidence=ev.rule(
                "question.verbatim_link.standalone",
                "The question names no parent metric, so it stands on its own — there "
                "is no metric whose score its answers would explain.",
                stage=3,
                inputs={"parent_question_id": None},
            ),
        )


class BlockIdTagger(QuestionTagger):
    name = "question.block_id"
    tag_dimension = "block_id"
    stage = 3
    source_type = "deterministic"

    def _tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        q = question
        if q.matrix_group_size and q.matrix_group_size > 1 and q.matrix_group_title:
            return TagResult(
                value=q.matrix_group_title, source="deterministic", confidence=1.0,
                evidence=ev.rule(
                    "question.block_id.matrix_member",
                    f"One of {q.matrix_group_size} rows sharing the matrix stem "
                    f'"{q.matrix_group_title}". Rows in a block share a scale and are '
                    "meant to be rendered as a single widget, not as separate charts.",
                    stage=3,
                    inputs={"matrix_group_title": q.matrix_group_title,
                            "matrix_group_size": q.matrix_group_size},
                ),
            )
        return TagResult(
            value=None, source="deterministic", status="skipped",
            evidence=ev.rule(
                "question.block_id.standalone",
                "The question is not part of a multi-row matrix block, so there is no "
                "group of siblings to render it with.",
                stage=3,
                inputs={"matrix_group_size": q.matrix_group_size or 0,
                        "matrix_group_title": q.matrix_group_title or "(none)"},
            ),
        )


def create_tagger() -> list[QuestionTagger]:
    return [DriverLinkTagger(), VerbatimLinkTagger(), BlockIdTagger()]
