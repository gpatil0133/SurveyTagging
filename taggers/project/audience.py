"""Audience type tagger: hybrid deterministic + LLM refinement."""

from models.context import UnifiedContext
from models.tags import TagAccumulator, TagResult
from taggers.base import ProjectTagger

# Keywords in question titles suggesting internal audience
_INTERNAL_KEYWORDS = [
    "employee", "workplace", "manager", "department", "team",
    "colleague", "supervisor", "HR", "work-life", "workload",
]

# Keywords suggesting external audience
_EXTERNAL_KEYWORDS = [
    "customer", "purchase", "product", "service", "store",
    "patient", "guest", "client", "buyer", "visit",
]


class AudienceTagger(ProjectTagger):
    name = "project.audience"
    tag_dimension = "audience_type"
    stage = 1
    depends_on = ["project.project_type"]
    source_type = "hybrid"

    def tag(self, context: UnifiedContext, accumulator: TagAccumulator) -> TagResult:
        survey_type = context.survey_meta.survey_type

        # Strong deterministic signals
        if survey_type == "EX":
            return TagResult(
                value="Internal \u2013 Employees",
                source="deterministic",
                confidence=0.95,
                evidence="surveyType=EX",
            )

        if survey_type == "CX":
            return TagResult(
                value="External \u2013 Customers",
                source="deterministic",
                confidence=0.80,
                evidence="surveyType=CX (tentative)",
            )

        if survey_type == "Assessment":
            return TagResult(
                value="Internal \u2013 Employees",
                source="deterministic",
                confidence=0.70,
                evidence="surveyType=Assessment",
            )

        # For "Survey" / "Poll" types — use heuristic signals
        purpose = context.overrides.purpose.lower()
        if "employee" in purpose:
            return TagResult(
                value="Internal \u2013 Employees",
                source="hybrid",
                confidence=0.80,
                evidence=f"Caller-supplied purpose: {context.overrides.purpose}",
            )

        # Check question content
        q_titles = " ".join(q.title.lower() for q in context.questions)
        internal_score = sum(1 for kw in _INTERNAL_KEYWORDS if kw in q_titles)
        external_score = sum(1 for kw in _EXTERNAL_KEYWORDS if kw in q_titles)

        if internal_score > external_score and internal_score >= 2:
            return TagResult(
                value="Internal \u2013 Employees",
                source="hybrid",
                confidence=0.75,
                evidence=f"Internal keywords ({internal_score}) > external ({external_score})",
            )
        elif external_score > internal_score and external_score >= 2:
            return TagResult(
                value="External \u2013 Customers",
                source="hybrid",
                confidence=0.75,
                evidence=f"External keywords ({external_score}) > internal ({internal_score})",
            )

        # Default — defer to LLM in stage 4 by setting low confidence
        return TagResult(
            value="External \u2013 Customers",
            source="hybrid",
            confidence=0.50,
            evidence="Insufficient signals for audience determination",
        )


def create_tagger() -> AudienceTagger:
    return AudienceTagger()
