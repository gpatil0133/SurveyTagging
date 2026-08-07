"""Flow dependency tagger: identifies dependency relationships between questions."""

from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger


class FlowDependencyTagger(QuestionTagger):
    name = "question.flow_dependency"
    tag_dimension = "flow_dependency"
    stage = 3
    source_type = "deterministic"

    def tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        if question.is_content_message:
            return TagResult(value=None, source="deterministic", status="skipped")

        # Conditionally Shown: follow-up question
        if question.is_followup_question:
            return TagResult(
                value="Conditionally Shown",
                source="deterministic",
                confidence=0.95,
                evidence=f"Follow-up to question {question.metric_question_id}",
            )

        # References Prior: has piping markers
        if question.has_piping_markers:
            return TagResult(
                value="References Prior",
                source="deterministic",
                confidence=0.90,
                evidence=f"Piping markers: {question.piping_markers[:2]}",
            )

        # Mutually Exclusive Block: part of matrix group with >1 member
        if question.matrix_group_size > 1 and question.matrix_group_title:
            return TagResult(
                value="Mutually Exclusive Block",
                source="deterministic",
                confidence=0.85,
                evidence=f"Matrix group: {question.matrix_group_title} ({question.matrix_group_size} rows)",
            )

        # Carries Forward: this question is referenced by follow-ups
        is_referenced = any(
            q.metric_question_id == question.question_id
            for q in context.questions
            if q.is_followup_question
        )
        if is_referenced:
            return TagResult(
                value="Carries Forward",
                source="deterministic",
                confidence=0.90,
                evidence="Referenced as metric question by follow-up(s)",
            )

        # Default: Independent
        return TagResult(
            value="Independent",
            source="deterministic",
            confidence=1.0,
        )


def create_tagger() -> FlowDependencyTagger:
    return FlowDependencyTagger()
