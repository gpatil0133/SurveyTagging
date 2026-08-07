"""Respondent sensitivity tagger: LLM-based with structural priors."""

from models import evidence as ev
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
            return TagResult(
                value=None, source="deterministic", status="skipped",
                evidence=ev.content_message("respondent_sensitivity", stage=5))

        role = accumulator.get_question_tag_value(question.question_id, "role_intent")

        # High Analytical Weight: primary metrics and key drivers
        if role == "Primary Metric" or question.is_nps or question.is_csat:
            return TagResult(
                value="High Analytical Weight",
                source="hybrid",
                confidence=0.90,
                evidence=ev.rule(
                    "question.respondent_sensitivity.headline_metric",
                    "This is a headline metric — role_intent is Primary Metric, or the "
                    "question is NPS or CSAT. Its answers carry disproportionate "
                    "analytical weight, so dropping or rewording it changes what the "
                    "whole survey reports.",
                    stage=5,
                    inputs={"role_intent": role or "(unset)",
                            "is_nps": question.is_nps,
                            "is_csat": question.is_csat},
                ),
            )

        if question.is_key_driver and role == "Driver / Attribute":
            return TagResult(
                value="High Analytical Weight",
                source="hybrid",
                confidence=0.85,
                evidence=ev.hybrid(
                    "question.respondent_sensitivity.key_driver",
                    "The platform marks this as a key driver and its role is "
                    "Driver / Attribute — it feeds the driver analysis behind the "
                    "headline metric, so it carries real analytical weight even though "
                    "it is not a headline number itself.",
                    components=[
                        ev.component("platform key-driver flag"),
                        ev.component("role_intent", "Driver / Attribute"),
                    ],
                    stage=5,
                ),
            )

        # Effort-intensive: large matrix groups or multiple consecutive open-ends
        if question.matrix_group_size > 8:
            return TagResult(
                value="Effort-intensive",
                source="hybrid",
                confidence=0.85,
                evidence=ev.statistic(
                    "question.respondent_sensitivity.large_matrix",
                    f"This row sits in a {question.matrix_group_size}-row matrix. "
                    "Grids past about eight rows are where respondents start "
                    "straight-lining and dropping out — the cost here is the burden on "
                    "the respondent, not the value of the answer.",
                    measure="matrix_group_size",
                    observed=question.matrix_group_size,
                    threshold=8,
                    stage=5,
                    inputs={"matrix_group_title": question.matrix_group_title or "(untitled)"},
                ),
            )

        # Optional / Low Stakes: standalone non-metric questions
        if role in ("Segmentation", "Contextual / Situational", "Profiling / Demographic"):
            return TagResult(
                value="Optional / Low Stakes",
                source="hybrid",
                confidence=0.70,
                evidence=ev.rule(
                    "question.respondent_sensitivity.context_role",
                    f"role_intent is {role} — the question describes or routes the "
                    "respondent rather than measuring anything, so it is cheap to "
                    "answer and nothing headline depends on it.",
                    stage=5,
                    inputs={"role_intent": role},
                ),
            )

        # Default — requires LLM for nuanced classification
        return TagResult(
            value="Optional / Low Stakes",
            source="llm",
            confidence=0.50,
            evidence=ev.fallback(
                "question.respondent_sensitivity.deferred_to_llm",
                f"No structural rule fired: role_intent is {role or 'unset'}, the "
                "question is not a headline metric or key driver, and it is not in a "
                "large grid. Judging respondent burden from wording alone needs the "
                "LLM, so this 0.50 placeholder is meant to be overwritten by Call 2.",
                stage=5,
                inputs={"role_intent": role or "(unset)",
                        "matrix_group_size": question.matrix_group_size},
            ),
        )


def create_tagger() -> RespondentSensitivityTagger:
    return RespondentSensitivityTagger()
