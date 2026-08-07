"""Project purpose tagger: LLM-based with strong structural priors."""

from models.context import UnifiedContext
from models.tags import TagAccumulator, TagResult
from taggers.base import ProjectTagger

# Title keyword → purpose mapping
_TITLE_PRIORS = [
    (["loyalty", "nps", "net promoter", "recommend"], "Measure Loyalty"),
    (["churn", "closure", "cancellation", "exit", "retention", "leaving"], "Understand Churn / Retention"),
    (["product feedback", "feature", "usability"], "Product / Service Feedback"),
    (["employee", "engagement", "workplace", "work culture"], "Employee Engagement"),
    (["onboarding", "training", "orientation", "learning"], "Onboarding / Training"),
    (["market research", "competitive", "concept test"], "Market Research / VOC"),
    (["compliance", "audit", "regulatory", "policy"], "Compliance / Audit"),
    (["event", "conference", "workshop", "seminar", "program"], "Event / Program Feedback"),
]

# Caller-supplied `purpose` override → purpose mapping
_OVERRIDE_PURPOSE_MAP = {
    "Product Feedback Surveys": "Product / Service Feedback",
    "Employee Surveys": "Employee Engagement",
    "Customer Satisfaction": "Measure Loyalty",
    "Market Research": "Market Research / VOC",
}


class PurposeTagger(ProjectTagger):
    name = "project.purpose"
    tag_dimension = "project_purpose"
    stage = 4
    depends_on = ["project.project_type", "project.audience"]
    source_type = "llm"

    def tag(self, context: UnifiedContext, accumulator: TagAccumulator) -> TagResult:
        # Prior 1: Survey type
        project_type = accumulator.get_project_tag_value("project_type")
        if project_type == "EX":
            return TagResult(
                value="Employee Engagement",
                source="hybrid",
                confidence=0.85,
                evidence="EX survey type → Employee Engagement",
            )

        # Prior 2: Title keyword matching (survey-specific, higher priority)
        title_lower = context.survey_meta.title.lower()
        for keywords, purpose in _TITLE_PRIORS:
            if any(kw in title_lower for kw in keywords):
                return TagResult(
                    value=purpose,
                    source="hybrid",
                    confidence=0.85,
                    evidence=f"Title matches {purpose} keywords",
                )

        # Prior 3: caller-supplied purpose override (tenant-level, lower specificity)
        override_purpose = context.overrides.purpose.strip()
        if override_purpose in _OVERRIDE_PURPOSE_MAP:
            return TagResult(
                value=_OVERRIDE_PURPOSE_MAP[override_purpose],
                source="hybrid",
                confidence=0.70,
                evidence=f"Caller-supplied purpose: {override_purpose}",
            )

        # Prior 4: Question content signals
        has_nps = context.has_nps
        has_csat = context.has_csat
        q_text = " ".join(q.title.lower() for q in context.non_cm_questions)

        if has_nps:
            return TagResult(
                value="Measure Loyalty",
                source="hybrid",
                confidence=0.75,
                evidence="Contains NPS question",
            )

        if has_csat:
            return TagResult(
                value="Product / Service Feedback",
                source="hybrid",
                confidence=0.70,
                evidence="Contains CSAT question",
            )

        if any(kw in q_text for kw in ["satisfaction", "feedback", "experience", "rate"]):
            return TagResult(
                value="Product / Service Feedback",
                source="hybrid",
                confidence=0.60,
                evidence="General satisfaction/feedback keywords in questions",
            )

        # Placeholder for LLM
        return TagResult(
            value="Product / Service Feedback",
            source="llm",
            confidence=0.40,
            evidence="Requires LLM classification",
        )


def create_tagger() -> PurposeTagger:
    return PurposeTagger()
