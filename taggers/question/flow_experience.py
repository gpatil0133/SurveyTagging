"""Flow respondent experience tagger: LLM-based with structural priors."""

from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger


class FlowExperienceTagger(QuestionTagger):
    name = "question.flow_experience"
    tag_dimension = "flow_respondent_experience"
    stage = 5
    depends_on = ["question.flow_placement"]
    source_type = "llm"

    def tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        if question.is_content_message:
            # CM at the start = Trust Builder (welcome/instructions)
            if question.position_index <= 1:
                title_lower = question.title.lower()
                if any(kw in title_lower for kw in ["welcome", "thank you", "this survey"]):
                    return TagResult(
                        value="Trust Builder",
                        source="hybrid",
                        confidence=0.85,
                        evidence="Welcome/intro content message",
                    )
            # Mid-survey CM = section header = Progress Anchor
            if question.position_index > 1:
                return TagResult(
                    value="Progress Anchor",
                    source="hybrid",
                    confidence=0.80,
                    evidence="Mid-survey section header",
                )
            return TagResult(value=None, source="deterministic", status="skipped")

        placement = accumulator.get_question_tag_value(question.question_id, "flow_placement")

        # Opening question = Trust Builder (easy entry point)
        if placement == "Opening":
            return TagResult(
                value="Trust Builder",
                source="hybrid",
                confidence=0.75,
                evidence="Opening question eases respondent in",
            )

        # Large matrix group = Effort Checkpoint
        if question.matrix_group_size > 6:
            return TagResult(
                value="Effort Checkpoint",
                source="hybrid",
                confidence=0.80,
                evidence=f"Matrix group with {question.matrix_group_size} rows",
            )

        # NPS/CSAT near end = Progress Anchor
        if (question.is_nps or question.is_csat) and placement in ("Deep Dive", "Closing"):
            return TagResult(
                value="Progress Anchor",
                source="hybrid",
                confidence=0.80,
                evidence="Key metric near end signals survey completion",
            )

        # Open-ended after rating block = Re-engagement Point
        if question.question_type == "T" and not question.is_followup_question:
            # Check if previous questions were all rating types
            idx = question.position_index
            if idx >= 3:
                prev_types = [
                    q.question_type for q in context.questions[max(0, idx - 3):idx]
                    if not q.is_content_message
                ]
                if all(t in ("RS", "RW", "GR", "RG", "RT") for t in prev_types if t):
                    return TagResult(
                        value="Re-engagement Point",
                        source="hybrid",
                        confidence=0.70,
                        evidence="Open-ended text after rating block",
                    )

        # Default — requires LLM
        return TagResult(
            value="Progress Anchor",
            source="llm",
            confidence=0.40,
            evidence="Requires LLM classification",
        )


def create_tagger() -> FlowExperienceTagger:
    return FlowExperienceTagger()
