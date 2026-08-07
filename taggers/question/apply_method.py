"""Apply method tagger: always System-applied for automated tags."""

from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger


class ApplyMethodTagger(QuestionTagger):
    name = "question.apply_method"
    tag_dimension = "apply_method"
    stage = 3
    source_type = "deterministic"

    def tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        return TagResult(
            value="System-applied",
            source="deterministic",
            confidence=1.0,
        )


def create_tagger() -> ApplyMethodTagger:
    return ApplyMethodTagger()
