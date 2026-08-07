"""visualization_type tagger — recommended chart type.

Stage 5, hybrid. Depends on metric_type, role_intent (Stage 3) + trend_trackable (Stage 4).
Per F1: no same-stage dependencies.
"""

from __future__ import annotations

from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger


class VisualizationTypeTagger(QuestionTagger):
    name = "question.visualization_type"
    tag_dimension = "visualization_type"
    stage = 5
    source_type = "hybrid"

    @property
    def depends_on(self) -> list[str]:
        return ["question.metric_type", "question.role_intent", "question.trend_trackable"]

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
        trend = accumulator.get_question_tag_value(q.question_id, "trend_trackable")

        # Open-ended → Word Cloud
        if metric_type == "Open-ended":
            return TagResult(value="Word Cloud", source="deterministic", confidence=0.95,
                             evidence="Open-ended text")

        # Primary Metric with Standard Metric → Score Card
        if role == "Primary Metric" and metric_type == "Standard Metric":
            # If trend-trackable + we had history → would use Line Chart; per F7,
            # we emit Score Card for per-survey projection and let needs_history flag
            # upgrade in Phase 4 aggregation.
            return TagResult(value="Score Card", source="deterministic", confidence=0.95,
                             evidence="Primary Metric + Standard Metric")

        # Driver in matrix → Heat Map
        if role == "Driver / Attribute" and q.matrix_group_title:
            return TagResult(value="Heat Map", source="deterministic", confidence=0.90,
                             evidence="Driver in matrix group")

        # Categorical
        if metric_type == "Categorical":
            n_opts = len(q.answer_options)
            if n_opts == 2:
                return TagResult(value="Pie Chart", source="deterministic", confidence=0.80,
                                 evidence="Binary choice")
            return TagResult(value="Bar Chart", source="deterministic", confidence=0.85,
                             evidence=f"{n_opts} categorical options")

        # Rating scale default
        if metric_type in ("Standard Metric", "Custom Metric"):
            return TagResult(value="Bar Chart", source="deterministic", confidence=0.75,
                             evidence="Rating scale default")

        # Fallback (LLM refines)
        return TagResult(value="Table", source="hybrid", confidence=0.40,
                         evidence=f"Fallback (type={metric_type})")


def create_tagger() -> VisualizationTypeTagger:
    return VisualizationTypeTagger()
