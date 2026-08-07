"""Project purpose tagger: LLM-based with strong structural priors."""

from models import evidence as ev
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
                evidence=ev.rule(
                    "project.purpose.ex_implies_engagement",
                    "The survey is typed EX. Employee-experience programmes are "
                    "overwhelmingly engagement studies, so that prior is taken before "
                    "any title or content inspection.",
                    stage=4,
                    inputs={"project_type": "EX"},
                ),
            )

        # Prior 2: Title keyword matching (survey-specific, higher priority)
        title_lower = context.survey_meta.title.lower()
        for keywords, purpose in _TITLE_PRIORS:
            if any(kw in title_lower for kw in keywords):
                return TagResult(
                    value=purpose,
                    source="hybrid",
                    confidence=0.85,
                    evidence=ev.rule(
                        "project.purpose.title_keyword",
                        f"The survey title names what the survey is for: it contains "
                        f"a keyword this tagger maps to {purpose}. A survey-specific "
                        "title outranks any tenant-level override below.",
                        stage=4,
                        inputs={"matched_keywords":
                                    [kw for kw in keywords if kw in title_lower],
                                "mapped_to": purpose},
                        quote=context.survey_meta.title,
                    ),
                )

        # Prior 3: caller-supplied purpose override (tenant-level, lower specificity)
        override_purpose = context.overrides.purpose.strip()
        if override_purpose in _OVERRIDE_PURPOSE_MAP:
            return TagResult(
                value=_OVERRIDE_PURPOSE_MAP[override_purpose],
                source="hybrid",
                confidence=0.70,
                evidence=ev.rule(
                    "project.purpose.caller_override",
                    f'The title said nothing usable, so the caller-supplied purpose '
                    f'"{override_purpose}" decides it. Ranked below the title because '
                    "it is a tenant-level setting, not a statement about this survey.",
                    stage=4,
                    inputs={"override": override_purpose,
                            "mapped_to": _OVERRIDE_PURPOSE_MAP[override_purpose]},
                ),
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
                evidence=ev.rule(
                    "project.purpose.has_nps",
                    "Neither the title nor an override said what this survey is for, "
                    "but it carries an NPS question — and NPS exists to measure "
                    "loyalty.",
                    stage=4,
                    inputs={"has_nps": True},
                ),
            )

        if has_csat:
            return TagResult(
                value="Product / Service Feedback",
                source="hybrid",
                confidence=0.70,
                evidence=ev.rule(
                    "project.purpose.has_csat",
                    "No title, override or NPS signal, but the survey carries a CSAT "
                    "question, which measures satisfaction with a product or service "
                    "rather than loyalty to the brand.",
                    stage=4,
                    inputs={"has_csat": True, "has_nps": False},
                ),
            )

        if any(kw in q_text for kw in ["satisfaction", "feedback", "experience", "rate"]):
            return TagResult(
                value="Product / Service Feedback",
                source="hybrid",
                confidence=0.60,
                evidence=ev.rule(
                    "project.purpose.generic_feedback_keywords",
                    "No structured metric to go on — just generic satisfaction / "
                    "feedback / experience wording in the questions. Enough to say "
                    "this is feedback collection, not enough to say what kind, hence "
                    "the low confidence.",
                    stage=4,
                    inputs={"has_nps": False, "has_csat": False},
                ),
            )

        # Placeholder for LLM
        return TagResult(
            value="Product / Service Feedback",
            source="llm",
            confidence=0.40,
            evidence=ev.fallback(
                "project.purpose.deferred_to_llm",
                "Every deterministic prior missed — no EX type, no title keyword, no "
                "caller override, no NPS or CSAT, no feedback vocabulary. The value is "
                "a placeholder held at 0.40 so LLM Call 1 overwrites it; if you are "
                "reading this in the output, that call did not run or did not answer.",
                stage=4,
            ),
        )


def create_tagger() -> PurposeTagger:
    return PurposeTagger()
