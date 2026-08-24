"""Role/intent tagger: hybrid deterministic (phase 1) + LLM (phase 2)."""

from __future__ import annotations

import re

from models import evidence as ev
from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers import _sub_types
from taggers.base import QuestionTagger

_DEMOGRAPHIC_PATTERNS = [
    re.compile(r"\bage\s*(group|range)?\b", re.I),
    re.compile(r"\bgender\b", re.I),
    re.compile(r"\bethnicity\b", re.I),
    re.compile(r"\beducation\s*level\b", re.I),
    re.compile(r"\bmarital\s*status\b", re.I),
    re.compile(r"\bdate\s*of\s*birth\b", re.I),
    re.compile(r"\bfull\s*name\b", re.I),
    re.compile(r"^name$", re.I),
    re.compile(r"\bemail\s*address\b", re.I),
    re.compile(r"\bcity\s*/?\s*town\b", re.I),
    re.compile(r"\bphone\s*(number)?\b", re.I),
]

_DEMOGRAPHIC_OPTION_KEYWORDS = [
    "male", "female", "non-binary", "prefer not",
    "18-24", "25-34", "35-44", "45-54", "55-64", "65",
    "employed", "unemployed", "retired", "student", "self-employed",
]


class RoleIntentTagger(QuestionTagger):
    name = "question.role_intent"
    tag_dimension = "role_intent"
    stage = 3
    source_type = "hybrid"

    def tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        if question.is_content_message:
            return TagResult(value=None, source="deterministic", status="skipped",
                             evidence=ev.content_message("role_intent", stage=3))

        q = question

        # Primary Metric: NPS / CSAT (CES is checked further down, after the
        # demographic and routing rules — see the rs_type==3 branch).
        _PRIMARY_RS = {2: ("NPS", "the 0-10 Net Promoter scale"),
                       4: ("CSAT", "the 5-point satisfaction scale")}
        if q.rs_type in _PRIMARY_RS:
            metric, scale = _PRIMARY_RS[q.rs_type]
            return TagResult(
                value="Primary Metric", source="deterministic", confidence=1.0,
                evidence=ev.rule(
                    "question.role_intent.standard_metric",
                    f"The platform flags this as {metric} ({scale}). Standard metrics "
                    "are what a survey is built around, so they take the primary role "
                    "before any other rule is considered.",
                    stage=3,
                    inputs={"rs_type": q.rs_type, "metric": metric},
                ),
            )

        # Primary Metric: Custom metric
        if q.is_custom_metric:
            return TagResult(
                value="Primary Metric", source="deterministic", confidence=0.95,
                evidence=ev.rule(
                    "question.role_intent.custom_metric",
                    "The survey author designated this a custom metric, so it is a "
                    "headline number for this tenant even though it is not one of the "
                    "standard benchmarkable scales.",
                    stage=3,
                    inputs={"is_custom_metric": True,
                            "custom_metric_title": q.custom_metric_title or "(untitled)"},
                ),
            )

        # Follow-up / Verbatim
        if q.is_followup_question and q.question_type == "T":
            return TagResult(
                value="Follow-up / Verbatim", source="deterministic", confidence=0.95,
                evidence=ev.hybrid(
                    "question.role_intent.followup_text",
                    f"A free-text question shown as a follow-up to question "
                    f"{q.metric_question_id} — the classic \"why did you score it that "
                    "way?\" verbatim.",
                    components=[
                        ev.component("question_type=T", "free text"),
                        ev.component(f"follow-up to {q.metric_question_id}",
                                     "conditionally shown after a metric"),
                    ],
                    stage=3,
                ),
            )

        # Profiling / Demographic: email subtype
        if q.question_type == "T" and q.question_sub_type in _sub_types.EMAIL:
            return TagResult(
                value="Profiling / Demographic", source="deterministic", confidence=0.95,
                evidence=ev.rule(
                    "question.role_intent.email_field",
                    "Sub-type 31 is the platform's email-validated text field. It "
                    "collects an identifier about the respondent rather than an "
                    "opinion.",
                    stage=3,
                    inputs={"question_type": "T",
                            "question_sub_type": q.question_sub_type},
                ),
            )

        # Profiling / Demographic: contact/signature
        if q.question_type == "CS":
            return TagResult(
                value="Profiling / Demographic", source="deterministic", confidence=0.95,
                evidence=ev.rule(
                    "question.role_intent.contact_block",
                    "Type CS is a contact-details or signature block — pure identity "
                    "capture, nothing measured.",
                    stage=3,
                    inputs={"question_type": "CS"},
                ),
            )

        # Profiling / Demographic: title matches demographic patterns
        if any(p.search(q.title) for p in _DEMOGRAPHIC_PATTERNS):
            return TagResult(
                value="Profiling / Demographic", source="deterministic", confidence=0.85,
                evidence=ev.rule(
                    "question.role_intent.demographic_title",
                    "The question wording matches a demographic pattern (age, gender, "
                    "ethnicity, education, name, contact details) — it describes who "
                    "the respondent is rather than what they think.",
                    stage=3,
                    inputs={"matched_patterns":
                                [p.pattern for p in _DEMOGRAPHIC_PATTERNS
                                 if p.search(q.title)][:3]},
                    quote=q.title,
                ),
            )

        # Profiling / Demographic: answer options are demographic
        if q.question_type in ("L", "R", "RT") and q.answer_options:
            opt_text = " ".join(o.answer_text.lower() for o in q.answer_options)
            demo_matches = sum(1 for kw in _DEMOGRAPHIC_OPTION_KEYWORDS if kw in opt_text)
            if demo_matches >= 3:
                return TagResult(
                    value="Profiling / Demographic", source="deterministic",
                    confidence=0.80,
                    evidence=ev.statistic(
                        "question.role_intent.demographic_options",
                        f"The wording gave nothing away, but {demo_matches} of the "
                        "answer options are demographic categories (male/female, age "
                        "bands, employment status). Three or more is treated as "
                        "conclusive; fewer could be coincidence.",
                        measure="demographic_option_matches",
                        observed=demo_matches,
                        threshold=3,
                        stage=3,
                        inputs={"question_type": q.question_type},
                    ),
                )

        # Contextual / Situational: date picker
        if q.question_type == "T" and q.question_sub_type in _sub_types.DATE:
            return TagResult(
                value="Contextual / Situational", source="deterministic", confidence=0.85,
                evidence=ev.rule(
                    "question.role_intent.date_picker",
                    "Sub-type 1 is a date picker. It records when something happened — "
                    "context for interpreting the other answers, not an opinion in "
                    "itself.",
                    stage=3,
                    inputs={"question_type": "T",
                            "question_sub_type": q.question_sub_type},
                ),
            )

        # Contextual / Situational: file upload
        if q.question_type == "T" and q.question_sub_type in _sub_types.FILE_UPLOAD:
            return TagResult(
                value="Contextual / Situational", source="deterministic", confidence=0.80,
                evidence=ev.rule(
                    "question.role_intent.file_upload",
                    "Sub-type 71 is a file upload — supporting evidence attached to the "
                    "response (a receipt, a photo), not a measurable answer.",
                    stage=3,
                    inputs={"question_type": "T",
                            "question_sub_type": q.question_sub_type},
                ),
            )

        # Segmentation: hidden radio (routing)
        if q.question_type == "HR":
            return TagResult(
                value="Segmentation", source="deterministic", confidence=0.85,
                evidence=ev.rule(
                    "question.role_intent.hidden_radio",
                    "A hidden radio is never shown to the respondent — the survey logic "
                    "sets it to record which path someone took, which makes it a "
                    "grouping variable.",
                    stage=3,
                    inputs={"question_type": "HR"},
                ),
            )

        # Primary Metric: CES (Customer Effort Score) — rs_type=3 per Sogolytics platform
        if q.rs_type == 3:
            return TagResult(
                value="Primary Metric", source="deterministic", confidence=1.0,
                evidence=ev.rule(
                    "question.role_intent.ces",
                    "The platform flags this as Customer Effort Score (rs_type=3), a "
                    "standard headline metric. Note this check sits below the "
                    "demographic and routing rules, so a CES-scaled question that also "
                    "looks demographic is classified as demographic.",
                    stage=3,
                    inputs={"rs_type": 3, "metric": "CES"},
                ),
            )

        # Driver/Attribute: Key driver flag
        if q.is_key_driver:
            return TagResult(
                value="Driver / Attribute", source="deterministic", confidence=0.90,
                evidence=ev.rule(
                    "question.role_intent.key_driver_flag",
                    "The survey author flagged this question as a key driver — it is "
                    "measured to explain movement in a headline metric rather than to "
                    "be reported on its own.",
                    stage=3,
                    inputs={"is_key_driver": True},
                ),
            )

        # Driver/Attribute: matrix/grid types
        if q.question_type in ("RW", "RK", "GR", "GC", "RG", "GQ"):
            return TagResult(
                value="Driver / Attribute", source="deterministic", confidence=0.80,
                evidence=ev.rule(
                    "question.role_intent.matrix_type",
                    f"Type {q.question_type} is a matrix, grid or ranking. Batteries "
                    "like this rate the attributes behind an outcome — they are "
                    "drivers by construction, even without the key-driver flag.",
                    stage=3,
                    inputs={"question_type": q.question_type},
                ),
            )

        # Segmentation: L/R/RT with categorical options (not rating)
        if q.question_type in ("L", "R", "SR", "ML", "C"):
            # If it has weights, it's more like a rating
            has_weights = any(o.weight is not None for o in q.answer_options)
            if not has_weights:
                return TagResult(
                    value="Segmentation", source="deterministic", confidence=0.70,
                    evidence=ev.rule(
                        "question.role_intent.unweighted_categorical",
                        f"Type {q.question_type} with unweighted options. Without "
                        "weights the answers cannot be scored or averaged — they are "
                        "labels, so the question's job is to group respondents.",
                        stage=3,
                        inputs={"question_type": q.question_type,
                                "has_weighted_options": False,
                                "option_count": len(q.answer_options)},
                    ),
                )

        # RT with weights = rating
        if q.question_type == "RT":
            return TagResult(
                value="Driver / Attribute", source="deterministic", confidence=0.75,
                evidence=ev.rule(
                    "question.role_intent.visual_rating",
                    "A visual rating question (stars, faces) whose options carry "
                    "weights, so it produces a score. Not flagged as a headline metric, "
                    "so it reads as an attribute rating.",
                    stage=3,
                    inputs={"question_type": "RT", "has_weighted_options": True},
                ),
            )

        # RS (generic rating scale)
        if q.question_type == "RS":
            return TagResult(
                value="Driver / Attribute", source="deterministic", confidence=0.75,
                evidence=ev.rule(
                    "question.role_intent.rating_scale",
                    "A generic rating scale with no standard-metric flag and no "
                    "key-driver flag. It measures one attribute on a scale, which is "
                    "the driver role.",
                    stage=3,
                    inputs={"question_type": "RS", "rs_type": q.rs_type,
                            "is_key_driver": False},
                ),
            )

        # Open-ended text: verbatim/follow-up or diagnostic
        if q.question_type == "T":
            # Check if it's at the end (closing question)
            if q.effective_position_ratio > 0.85:
                return TagResult(
                    value="Follow-up / Verbatim", source="hybrid", confidence=0.70,
                    evidence=ev.statistic(
                        "question.role_intent.closing_text",
                        f"A standalone open-end sitting "
                        f"{q.effective_position_ratio:.0%} of the way through the "
                        "survey. Text boxes in the final stretch are almost always the "
                        '"anything else you\'d like to tell us?" catch-all.',
                        measure="position_ratio",
                        observed=round(q.effective_position_ratio, 2),
                        threshold=0.85,
                        stage=3,
                        inputs={"question_type": "T"},
                    ),
                )
            return TagResult(
                value="Diagnostic", source="hybrid", confidence=0.60,
                evidence=ev.fallback(
                    "question.role_intent.midsurvey_text",
                    f"A standalone open-end "
                    f"{q.effective_position_ratio:.0%} of the way through — too early "
                    "to be the closing catch-all, and not attached to a metric. "
                    "Diagnostic is the best structural guess; only the wording can "
                    "settle it, which is why the LLM pass may refine this.",
                    stage=3,
                    inputs={"question_type": "T",
                            "position_ratio": round(q.effective_position_ratio, 2)},
                ),
            )

        # Fallback
        return TagResult(
            value="Segmentation", source="hybrid", confidence=0.50,
            evidence=ev.fallback(
                "question.role_intent.no_rule_matched",
                f"None of the rules matched question type {q.question_type!r} "
                f"(rs_type={q.rs_type}). Segmentation is the placeholder; the 0.50 is "
                "low enough that LLM Call 2 will overwrite it. A type reaching this "
                "line repeatedly means a rule is missing above.",
                stage=3,
                inputs={"question_type": q.question_type or "(empty)",
                        "rs_type": q.rs_type,
                        "question_sub_type": q.question_sub_type},
            ),
        )


def create_tagger() -> RoleIntentTagger:
    return RoleIntentTagger()
