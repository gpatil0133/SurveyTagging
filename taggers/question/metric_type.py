"""metric_type tagger — classifies the question's measurement type.

Stage 3, deterministic. No accumulator dependencies.
"""

from __future__ import annotations

from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger


class MetricTypeTagger(QuestionTagger):
    name = "question.metric_type"
    tag_dimension = "metric_type"
    stage = 3
    source_type = "deterministic"

    def tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        q = question

        if q.is_content_message:
            return TagResult(value=None, source="deterministic", status="skipped",
                             evidence="Content message")

        # Standard metrics by rs_type
        if q.rs_type == 2:
            return TagResult(value="Standard Metric", source="deterministic",
                             confidence=1.0, evidence="rs_type=2 (NPS)")
        if q.rs_type == 3:
            return TagResult(value="Standard Metric", source="deterministic",
                             confidence=1.0, evidence="rs_type=3 (CES)")
        if q.rs_type == 4:
            return TagResult(value="Standard Metric", source="deterministic",
                             confidence=1.0, evidence="rs_type=4 (CSAT)")

        # Custom metrics (platform-flagged)
        if q.is_custom_metric:
            return TagResult(value="Custom Metric", source="deterministic",
                             confidence=1.0,
                             evidence=f"is_custom_metric=true ({q.custom_metric_title})")

        # Grid/matrix question types — custom metrics
        if q.question_type in ("GR", "GC", "RG", "GQ"):
            return TagResult(value="Custom Metric", source="deterministic",
                             confidence=0.90,
                             evidence=f"Grid/matrix type {q.question_type}")

        # Rating scales with weighted answers — custom metric
        if q.question_type in ("RS", "RT", "RW", "RK"):
            has_weights = any(o.weight is not None for o in q.answer_options)
            if has_weights:
                return TagResult(value="Custom Metric", source="deterministic",
                                 confidence=0.85,
                                 evidence=f"Rating scale {q.question_type} with weights")

        # Open-ended text
        if q.question_type == "T":
            return TagResult(value="Open-ended", source="deterministic",
                             confidence=1.0, evidence="Text question (T)")

        # Categorical — selection questions without numeric weights
        if q.question_type in ("L", "R", "C", "HR", "ML", "SR"):
            return TagResult(value="Categorical", source="deterministic",
                             confidence=0.95,
                             evidence=f"Selection type {q.question_type}")

        # Contact/Signature — treat as not-a-metric
        if q.question_type == "CS":
            return TagResult(value="Not Applicable", source="deterministic",
                             confidence=1.0, evidence="Contact/signature question")

        # Fallback
        return TagResult(value="Not Applicable", source="deterministic",
                         confidence=0.50, evidence=f"Unclassified type {q.question_type}")


def create_tagger() -> MetricTypeTagger:
    return MetricTypeTagger()
