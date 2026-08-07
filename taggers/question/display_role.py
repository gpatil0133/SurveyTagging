"""display_role tagger — how the question is displayed on dashboards.

Stage 5, hybrid. Depends on metric_type, role_intent (Stage 3), trend_trackable (Stage 4).

display_role depends on dashboard_placement (both Stage 5). Per F1: alphabetical
"da" < "di", so dashboard_placement runs BEFORE display_role within Stage 5. ✓
"""

from __future__ import annotations

from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger


class DisplayRoleTagger(QuestionTagger):
    name = "question.display_role"
    tag_dimension = "display_role"
    stage = 5
    source_type = "hybrid"

    @property
    def depends_on(self) -> list[str]:
        return ["question.metric_type", "question.role_intent", "question.trend_trackable",
                "question.dashboard_placement"]

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

        if role == "Primary Metric" and metric_type == "Standard Metric":
            return TagResult(value="Primary KPI", source="deterministic", confidence=0.95,
                             evidence="Primary Metric + Standard")

        if role == "Benchmarkable":
            return TagResult(value="Comparison Metric", source="deterministic", confidence=0.90,
                             evidence="role_intent=Benchmarkable")

        if role == "Driver / Attribute":
            return TagResult(value="Supporting Metric", source="deterministic", confidence=0.85,
                             evidence="role_intent=Driver/Attribute")

        if role == "Follow-up / Verbatim":
            return TagResult(value="Detail View", source="deterministic", confidence=0.90,
                             evidence="role_intent=Follow-up/Verbatim")

        if role == "Diagnostic":
            return TagResult(value="Drill-down", source="deterministic", confidence=0.85,
                             evidence="role_intent=Diagnostic")

        if role == "Primary Metric":
            return TagResult(value="Primary KPI", source="deterministic", confidence=0.80,
                             evidence="Primary Metric (non-standard)")

        if trend == "Yes":
            return TagResult(value="Trend Indicator", source="hybrid", confidence=0.65,
                             evidence="trend_trackable=Yes")

        return TagResult(value="Detail View", source="hybrid", confidence=0.45,
                         evidence=f"Default fallback (role={role})")


def create_tagger() -> DisplayRoleTagger:
    return DisplayRoleTagger()
