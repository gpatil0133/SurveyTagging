"""Flow placement tagger: position-based classification within survey flow."""

from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger


class FlowPlacementTagger(QuestionTagger):
    name = "question.flow_placement"
    tag_dimension = "flow_placement"
    stage = 3
    source_type = "deterministic"

    def tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        if question.is_content_message:
            return TagResult(value=None, source="deterministic", status="skipped",
                             evidence="Content message")

        total_non_cm = len(context.non_cm_questions)

        if total_non_cm == 0:
            return TagResult(value=None, source="deterministic", status="skipped")

        if total_non_cm == 1:
            return TagResult(value="Standalone", source="deterministic", confidence=1.0)

        ratio = question.effective_position_ratio

        if ratio == 0.0:
            return TagResult(value="Opening", source="deterministic", confidence=1.0,
                             evidence="First non-CM question")

        if ratio <= 0.15:
            return TagResult(value="Warm-up", source="deterministic", confidence=0.90,
                             evidence=f"Position ratio {ratio:.2f}")

        if ratio <= 0.75:
            return TagResult(value="Core Block", source="deterministic", confidence=0.90,
                             evidence=f"Position ratio {ratio:.2f}")

        if ratio <= 0.90:
            return TagResult(value="Deep Dive", source="deterministic", confidence=0.85,
                             evidence=f"Position ratio {ratio:.2f}")

        return TagResult(value="Closing", source="deterministic", confidence=0.90,
                         evidence=f"Position ratio {ratio:.2f}")


def create_tagger() -> FlowPlacementTagger:
    return FlowPlacementTagger()
