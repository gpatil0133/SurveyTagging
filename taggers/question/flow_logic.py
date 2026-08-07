"""Flow logic role tagger: identifies branching, piping, and routing roles."""

from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger


class FlowLogicTagger(QuestionTagger):
    name = "question.flow_logic"
    tag_dimension = "flow_logic_role"
    stage = 3
    source_type = "deterministic"

    def tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        if question.is_content_message:
            return TagResult(value=[], source="deterministic", status="skipped")

        roles: list[str] = []

        # Branching Trigger: hidden radio (routing question)
        if question.question_type == "HR":
            roles.append("Branching Trigger")

        # Branching Target: follow-up question (conditionally shown)
        if question.is_followup_question:
            roles.append("Branching Target")

        # Piping Target: question contains piping markers
        if question.has_piping_markers:
            roles.append("Piping Target")

        # Piping Source: this question's ID is referenced as metricQuestion by another
        is_referenced = any(
            q.metric_question_id == question.question_id
            for q in context.questions
            if q.is_followup_question
        )
        if is_referenced:
            roles.append("Piping Source")

        # Termination Trigger: Yes/No radio at position ≤1
        if (
            question.question_type == "R"
            and question.effective_position_ratio <= 0.1
            and len(question.answer_options) == 2
        ):
            opt_texts = {o.answer_text.lower().strip() for o in question.answer_options}
            if opt_texts == {"yes", "no"}:
                roles.append("Termination Trigger")

        return TagResult(
            value=roles,
            source="deterministic",
            confidence=0.90 if roles else 1.0,
        )


def create_tagger() -> FlowLogicTagger:
    return FlowLogicTagger()
