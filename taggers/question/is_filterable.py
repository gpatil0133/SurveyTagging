"""is_filterable tagger — whether the question works as a filter/facet in reports.

Stage 3, deterministic. No tag dependencies.
"""

from __future__ import annotations

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
                             evidence="Content message")

        if q.question_type in _FILTERABLE_TYPES:
            return TagResult(value="Yes", source="deterministic", confidence=1.0,
                             evidence=f"Categorical type {q.question_type}")

        # Explicit No for text/rating/grid/contact/signature
        return TagResult(value="No", source="deterministic", confidence=1.0,
                         evidence=f"Non-categorical type {q.question_type}")


def create_tagger() -> IsFilterableTagger:
    return IsFilterableTagger()
