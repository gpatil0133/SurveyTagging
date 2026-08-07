"""Respondent sensitivity tagger: LLM-based with structural priors."""

from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger


class RespondentSensitivityTagger(QuestionTagger):
    name = "question.respondent_sensitivity"
    tag_dimension = "respondent_sensitivity"
    stage = 5
    depends_on = ["question.role_intent"]
    source_type = "llm"

    def tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        if question.is_content_message:
            return TagResult(value=None, source="deterministic", status="skipped")

        role = accumulator.get_question_tag_value(question.question_id, "role_intent")

        # High Analytical Weight: primary metrics and key drivers
        if role == "Primary Metric" or question.is_nps or question.is_csat:
            return TagResult(
                value="High Analytical Weight",
                source="hybrid",
                confidence=0.90,
                evidence="Primary metric / NPS / CSAT question",
            )

        if question.is_key_driver and role == "Driver / Attribute":
            return TagResult(
                value="High Analytical Weight",
                source="hybrid",
                confidence=0.85,
                evidence="Key driver question",
            )

        # Effort-intensive: large matrix groups or multiple consecutive open-ends
        if question.matrix_group_size > 8:
            return TagResult(
                value="Effort-intensive",
                source="hybrid",
                confidence=0.85,
                evidence=f"Large matrix group ({question.matrix_group_size} rows)",
            )

        # Optional / Low Stakes: standalone non-metric questions
        if role in ("Segmentation", "Contextual / Situational", "Profiling / Demographic"):
            return TagResult(
                value="Optional / Low Stakes",
                source="hybrid",
                confidence=0.70,
                evidence=f"Non-metric role: {role}",
            )

        # Default — requires LLM for nuanced classification
        return TagResult(
            value="Optional / Low Stakes",
            source="llm",
            confidence=0.50,
            evidence="Requires LLM classification",
        )


def create_tagger() -> RespondentSensitivityTagger:
    return RespondentSensitivityTagger()
