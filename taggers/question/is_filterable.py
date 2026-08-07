"""is_filterable tagger — whether the question works as a filter/facet in reports.

Stage 3, deterministic. No tag dependencies.
"""

from __future__ import annotations

from models import evidence as ev
from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger


_FILTERABLE_TYPES = {"L", "R", "C", "HR", "ML", "SR"}


class IsFilterableTagger(QuestionTagger):
    name = "question.is_filterable"
    tag_dimension = "is_filterable"
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
                             evidence=ev.content_message("is_filterable", stage=3))

        if q.question_type in _FILTERABLE_TYPES:
            return TagResult(
                value="Yes", source="deterministic", confidence=1.0,
                evidence=ev.rule(
                    "question.is_filterable.categorical_type",
                    f"Question type {q.question_type} produces a bounded set of discrete "
                    "answer choices, which is exactly what a report filter needs to "
                    "build a facet from.",
                    stage=3,
                    inputs={"question_type": q.question_type},
                ),
            )

        # Explicit No for text/rating/grid/contact/signature
        return TagResult(
            value="No", source="deterministic", confidence=1.0,
            evidence=ev.rule(
                "question.is_filterable.non_categorical_type",
                f"Question type {q.question_type} is free text, a rating, a grid, a "
                "contact block or a signature — none of which yields the discrete, "
                "bounded answer set a filter facet requires.",
                stage=3,
                inputs={"question_type": q.question_type,
                        "filterable_types": sorted(_FILTERABLE_TYPES)},
            ),
        )


def create_tagger() -> IsFilterableTagger:
    return IsFilterableTagger()
