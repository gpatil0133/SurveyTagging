"""metric_type tagger — classifies the question's measurement type.

Stage 3, deterministic. No accumulator dependencies.
"""

from __future__ import annotations

from models import evidence as ev
from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger


class MetricTypeTagger(QuestionTagger):
    name = "question.metric_type"
    tag_dimension = "metric_type"
    stage = 3
    source_type = "deterministic"

    def _tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        q = question

        # Standard metrics by rs_type
        _STANDARD = {2: ("NPS", "Net Promoter Score"),
                     3: ("CES", "Customer Effort Score"),
                     4: ("CSAT", "Customer Satisfaction")}
        if q.rs_type in _STANDARD:
            short, full = _STANDARD[q.rs_type]
            return TagResult(
                value="Standard Metric", source="deterministic", confidence=1.0,
                evidence=ev.rule(
                    "question.metric_type.standard_rs_type",
                    f"The platform itself flags this as a {full} ({short}) question via "
                    f"rs_type={q.rs_type}. Standard metrics are benchmarkable across "
                    "tenants, which is what separates them from custom ones.",
                    stage=3,
                    inputs={"rs_type": q.rs_type, "metric": short},
                ),
            )

        # Custom metrics (platform-flagged)
        if q.is_custom_metric:
            return TagResult(
                value="Custom Metric", source="deterministic", confidence=1.0,
                evidence=ev.rule(
                    "question.metric_type.platform_custom_metric",
                    "The platform flags this question as a customer-defined metric. It "
                    "is scored and trended like a standard metric but is not "
                    "comparable across tenants.",
                    stage=3,
                    inputs={"is_custom_metric": True,
                            "custom_metric_title": q.custom_metric_title or "(untitled)"},
                ),
            )

        # Grid/matrix question types — custom metrics
        if q.question_type in ("GR", "GC", "RG", "GQ"):
            return TagResult(
                value="Custom Metric", source="deterministic", confidence=0.90,
                evidence=ev.rule(
                    "question.metric_type.grid_inferred",
                    f"Type {q.question_type} is a grid/matrix, which almost always "
                    "holds a battery of rated statements — a custom metric in "
                    "practice, though the platform did not flag it as one. Inferred "
                    "from structure, hence 0.90 rather than 1.0.",
                    stage=3,
                    inputs={"question_type": q.question_type,
                            "is_custom_metric": False},
                ),
            )

        # Rating scales with weighted answers — custom metric
        if q.question_type in ("RS", "RT", "RW", "RK"):
            has_weights = any(o.weight is not None for o in q.answer_options)
            if has_weights:
                return TagResult(
                    value="Custom Metric", source="deterministic", confidence=0.85,
                    evidence=ev.rule(
                        "question.metric_type.weighted_scale_inferred",
                        f"Type {q.question_type} is a rating scale and its answer "
                        "options carry numeric weights, so responses roll up to a "
                        "score. Weights are what make it a metric rather than a "
                        "categorical pick.",
                        stage=3,
                        inputs={"question_type": q.question_type,
                                "has_weighted_options": True,
                                "option_count": len(q.answer_options)},
                    ),
                )

        # Open-ended text
        if q.question_type == "T":
            return TagResult(
                value="Open-ended", source="deterministic", confidence=1.0,
                evidence=ev.rule(
                    "question.metric_type.text",
                    "Type T is a free-text question — the answer is prose, so there is "
                    "nothing to score numerically.",
                    stage=3,
                    inputs={"question_type": "T"},
                ),
            )

        # Categorical — selection questions without numeric weights
        if q.question_type in ("L", "R", "C", "HR", "ML", "SR"):
            return TagResult(
                value="Categorical", source="deterministic", confidence=0.95,
                evidence=ev.rule(
                    "question.metric_type.selection_type",
                    f"Type {q.question_type} asks the respondent to pick from a list "
                    "and its options carry no weights, so answers are categories to be "
                    "counted rather than values to be averaged.",
                    stage=3,
                    inputs={"question_type": q.question_type,
                            "option_count": len(q.answer_options)},
                ),
            )

        # Contact/Signature — treat as not-a-metric
        if q.question_type == "CS":
            return TagResult(
                value="Not Applicable", source="deterministic", confidence=1.0,
                evidence=ev.rule(
                    "question.metric_type.contact_signature",
                    "Type CS is a contact-details or signature block. It captures "
                    "identity, not an opinion, so no measurement type applies.",
                    stage=3,
                    inputs={"question_type": "CS"},
                ),
            )

        # Fallback
        return TagResult(
            value="Not Applicable", source="deterministic", confidence=0.50,
            evidence=ev.fallback(
                "question.metric_type.unknown_type",
                f"Question type {q.question_type!r} is not one this tagger recognizes, "
                "so no measurement type could be determined. The 0.50 confidence marks "
                "this as unclassified rather than as a finding of Not Applicable — a "
                "new platform question type probably needs adding here.",
                stage=3,
                inputs={"question_type": q.question_type or "(empty)"},
            ),
        )


def create_tagger() -> MetricTypeTagger:
    return MetricTypeTagger()
