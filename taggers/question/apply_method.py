"""Apply method tagger: always System-applied for automated tags."""

from models import evidence as ev
from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger


class ApplyMethodTagger(QuestionTagger):
    name = "question.apply_method"
    tag_dimension = "apply_method"
    stage = 3
    source_type = "deterministic"
    # Applies to every row the pipeline writes, content messages included:
    # "who assigned this tag" has the same answer for a CM as for a question.
    skips_content_messages = False

    def _tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        return TagResult(
            value="System-applied",
            source="deterministic",
            confidence=1.0,
            evidence=ev.rule(
                "question.apply_method.always_system",
                "Every tag this pipeline writes is machine-assigned, so apply_method "
                "is System-applied by construction. It flips to User-applied only when "
                "a human overrides a tag downstream — it is not an inference about the "
                "question.",
                stage=3,
            ),
        )


def create_tagger() -> ApplyMethodTagger:
    return ApplyMethodTagger()
