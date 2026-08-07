"""trend_trackable tagger — whether the question should be tracked over time.

Stage 4, deterministic. Depends on metric_type + role_intent.
"""

from __future__ import annotations

from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger


class TrendTrackableTagger(QuestionTagger):
    name = "question.trend_trackable"
    tag_dimension = "trend_trackable"
    stage = 4
    source_type = "deterministic"

    @property
    def depends_on(self) -> list[str]:
        return ["question.metric_type", "question.role_intent"]

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

        metric_type = accumulator.get_question_tag_value(q.question_id, "metric_type")
        role = accumulator.get_question_tag_value(q.question_id, "role_intent")

        # Standard Metrics always track
        if metric_type == "Standard Metric":
            return TagResult(value="Yes", source="deterministic", confidence=1.0,
                             evidence="Standard Metric (NPS/CSAT/CES/eNPS)")

        # Benchmarkable or Primary metric — track
        if role in ("Primary Metric", "Benchmarkable"):
            return TagResult(value="Yes", source="deterministic", confidence=0.90,
                             evidence=f"role_intent={role}")

        # Text/verbatim → no trending on raw text
        if metric_type == "Open-ended":
            return TagResult(value="No", source="deterministic", confidence=1.0,
                             evidence="Open-ended text cannot trend")

        # Segmentation / demographics — don't trend
        if role in ("Segmentation", "Profiling / Demographic", "Screener"):
            return TagResult(value="No", source="deterministic", confidence=0.95,
                             evidence=f"role_intent={role} is not a trend metric")

        # Drivers on known metric — track as supporting
        if role == "Driver / Attribute" and metric_type in ("Standard Metric", "Custom Metric"):
            return TagResult(value="Yes", source="deterministic", confidence=0.70,
                             evidence="Driver metric with numeric scale")

        return TagResult(value="No", source="deterministic", confidence=0.65,
                         evidence=f"Default: not trend-trackable (type={metric_type}, role={role})")


def create_tagger() -> TrendTrackableTagger:
    return TrendTrackableTagger()
