"""calculation_type tagger — how the question's responses should be aggregated.

Stage 4, deterministic. Depends on metric_type + metric_name (both Stage 3).
"""

from __future__ import annotations

from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger


class CalculationTypeTagger(QuestionTagger):
    name = "question.calculation_type"
    tag_dimension = "calculation_type"
    stage = 4
    source_type = "deterministic"

    @property
    def depends_on(self) -> list[str]:
        return ["question.metric_type", "question.metric_name"]

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

        metric_name = accumulator.get_question_tag_value(q.question_id, "metric_name")
        metric_type = accumulator.get_question_tag_value(q.question_id, "metric_type")

        # Platform source hint — trust if non-empty
        if q.calculation_type:
            # Normalize common Sogolytics names to our enum
            ct = q.calculation_type.strip().lower()
            if "weighted" in ct or "mean" in ct or "average" in ct:
                return TagResult(value="Mean", source="deterministic", confidence=1.0,
                                 evidence=f"Platform calculationType: {q.calculation_type}")
            if "percentage" in ct or "percent" in ct or ct == "%":
                return TagResult(value="Percentage", source="deterministic",
                                 confidence=1.0, evidence=f"Platform: {q.calculation_type}")
            if "nps" in ct:
                return TagResult(value="NPS Score", source="deterministic",
                                 confidence=1.0, evidence=f"Platform: {q.calculation_type}")
            if "sum" in ct:
                return TagResult(value="Sum", source="deterministic", confidence=1.0,
                                 evidence=f"Platform: {q.calculation_type}")

        # Named standard metrics
        if metric_name in ("NPS", "eNPS"):
            return TagResult(value="NPS Score", source="deterministic", confidence=1.0,
                             evidence=f"metric_name={metric_name}")
        if metric_name == "CSAT":
            return TagResult(value="Mean", source="deterministic", confidence=0.95,
                             evidence="metric_name=CSAT")
        if metric_name == "CES":
            return TagResult(value="Mean", source="deterministic", confidence=0.95,
                             evidence="metric_name=CES")

        # Ranking
        if q.question_type in ("RW", "RK"):
            return TagResult(value="Mean Rank", source="deterministic", confidence=0.95,
                             evidence=f"Ranking question type {q.question_type}")

        # By metric_type + options count
        if metric_type == "Open-ended":
            return TagResult(value="Count", source="deterministic", confidence=0.95,
                             evidence="Open-ended text")

        if metric_type == "Categorical":
            n_opts = len(q.answer_options)
            if n_opts == 2:
                return TagResult(value="Percentage", source="deterministic",
                                 confidence=0.90, evidence="Binary choice (2 options)")
            if n_opts >= 3:
                return TagResult(value="Distribution", source="deterministic",
                                 confidence=0.85, evidence=f"{n_opts} categorical options")

        if metric_type in ("Standard Metric", "Custom Metric"):
            # Rating scales default to Mean
            return TagResult(value="Mean", source="deterministic", confidence=0.80,
                             evidence=f"metric_type={metric_type}")

        # Not applicable
        return TagResult(value="Not Applicable", source="deterministic",
                         confidence=0.70,
                         evidence=f"No calculation applicable for type {q.question_type}")


def create_tagger() -> CalculationTypeTagger:
    return CalculationTypeTagger()
