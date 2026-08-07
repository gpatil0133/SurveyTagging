"""Audience type tagger: hybrid deterministic + LLM refinement."""

from models import evidence as ev
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
                evidence=ev.rule(
                    "project.audience.survey_type_ex",
                    "The platform classifies this as an EX survey, and EX means the "
                    "respondents are the organisation's own employees.",
                    stage=1,
                    inputs={"survey_type": "EX"},
                ),
            )

        if survey_type == "CX":
            return TagResult(
                value="External \u2013 Customers",
                source="deterministic",
                confidence=0.80,
                evidence=ev.rule(
                    "project.audience.survey_type_cx",
                    "The platform classifies this as a CX survey, so the respondents "
                    "are customers. Held at 0.80 rather than 0.95 on purpose: CX "
                    "surveys are sometimes fielded to partners or internal proxies, so "
                    "the LLM pass is left free to overturn this.",
                    stage=1,
                    inputs={"survey_type": "CX"},
                ),
            )

        if survey_type == "Assessment":
            return TagResult(
                value="Internal \u2013 Employees",
                source="deterministic",
                confidence=0.70,
                evidence=ev.rule(
                    "project.audience.survey_type_assessment",
                    "Assessments are usually fielded internally (skills, compliance, "
                    "360s), so employees are the default audience — but assessments "
                    "also go to candidates and students, hence the modest confidence.",
                    stage=1,
                    inputs={"survey_type": "Assessment"},
                ),
            )

        # For "Survey" / "Poll" types — use heuristic signals
        purpose = context.overrides.purpose.lower()
        if "employee" in purpose:
            return TagResult(
                value="Internal \u2013 Employees",
                source="hybrid",
                confidence=0.80,
                evidence=ev.rule(
                    "project.audience.override_purpose_employee",
                    "The platform type is generic, but the caller-supplied purpose "
                    'mentions "employee" — a stated purpose beats keyword counting.',
                    stage=1,
                    inputs={"survey_type": survey_type or "(unset)"},
                    quote=context.overrides.purpose,
                ),
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
                evidence=ev.statistic(
                    "project.audience.keyword_lean_internal",
                    f"The platform type is generic, so the question wording decides it: "
                    f"{internal_score} workplace keyword(s) (employee, manager, "
                    f"workload...) against {external_score} customer-facing ones.",
                    measure="internal_keyword_hits",
                    observed=internal_score,
                    threshold=2,
                    stage=1,
                    inputs={"external_keyword_hits": external_score,
                            "survey_type": survey_type or "(unset)"},
                ),
            )
        elif external_score > internal_score and external_score >= 2:
            return TagResult(
                value="External \u2013 Customers",
                source="hybrid",
                confidence=0.75,
                evidence=ev.statistic(
                    "project.audience.keyword_lean_external",
                    f"The platform type is generic, so the question wording decides it: "
                    f"{external_score} customer-facing keyword(s) (customer, purchase, "
                    f"patient, guest...) against {internal_score} workplace ones.",
                    measure="external_keyword_hits",
                    observed=external_score,
                    threshold=2,
                    stage=1,
                    inputs={"internal_keyword_hits": internal_score,
                            "survey_type": survey_type or "(unset)"},
                ),
            )

        # Default — defer to LLM in stage 4 by setting low confidence
        return TagResult(
            value="External \u2013 Customers",
            source="hybrid",
            confidence=0.50,
            evidence=ev.fallback(
                "project.audience.no_signal",
                f"Nothing decided this: the platform type is generic, no caller purpose "
                f"was supplied, and the question wording gave {internal_score} internal "
                f"vs {external_score} external keyword hits — too few, or too close, to "
                "call. External – Customers is the default, and the 0.50 confidence is "
                "the signal for the Stage 4 LLM pass to overwrite it.",
                stage=1,
                inputs={"survey_type": survey_type or "(unset)",
                        "internal_keyword_hits": internal_score,
                        "external_keyword_hits": external_score},
            ),
        )


def create_tagger() -> AudienceTagger:
    return AudienceTagger()
