"""dashboard_placement tagger — which dashboards this question appears on.

Stage 5, multi-label, hybrid+LLM. Uses only Stage 1-4 tags
(per audit F1 — no same-stage dependencies).

Constrains values to project-level `dashboard_routing` (which is set at Stage 4).
"""

from __future__ import annotations

from models.context import UnifiedContext
from models.survey import QuestionContext
from models.tags import TagAccumulator, TagResult
from taggers.base import QuestionTagger


class DashboardPlacementTagger(QuestionTagger):
    name = "question.dashboard_placement"
    tag_dimension = "dashboard_placement"
    stage = 5
    source_type = "hybrid"

    @property
    def depends_on(self) -> list[str]:
        return ["question.metric_type", "question.role_intent", "project.dashboard_routing"]

    def tag_question(
        self,
        context: UnifiedContext,
        question: QuestionContext,
        accumulator: TagAccumulator,
    ) -> TagResult:
        q = question

        if q.is_content_message:
            return TagResult(value=[], source="deterministic", status="skipped",
                             evidence="Content message")

        metric_type = accumulator.get_question_tag_value(q.question_id, "metric_type")
        role = accumulator.get_question_tag_value(q.question_id, "role_intent")
        project_type = accumulator.get_project_tag_value("project_type")
        project_dashboards = accumulator.get_project_tag_value("dashboard_routing") or []
        if not isinstance(project_dashboards, list):
            project_dashboards = [project_dashboards]

        placements: list[str] = []

        def maybe_add(name: str) -> None:
            if name in project_dashboards and name not in placements:
                placements.append(name)

        # Standard Metric (NPS/CSAT/CES/eNPS) + Primary Metric → Executive Dashboard
        if metric_type == "Standard Metric" and role == "Primary Metric":
            maybe_add("Executive Dashboard")

        # Open-ended text → VoC/VoE dashboards
        if metric_type == "Open-ended":
            if project_type == "CX":
                maybe_add("Voice of Customer Dashboard")
            elif project_type == "EX":
                maybe_add("Voice of Employee Dashboard")

        # Drivers → project_type-specific dashboard
        if role == "Driver / Attribute":
            if project_type == "CX":
                maybe_add("Customer Experience Dashboard")
            elif project_type == "EX":
                maybe_add("Employee Engagement Dashboard")

        # If nothing matched, default to first project-level dashboard so every
        # question has at least one placement (LLM refines).
        if not placements and project_dashboards:
            placements = [project_dashboards[0]]

        return TagResult(value=placements, source="hybrid", confidence=0.60,
                         evidence=f"Rules-based from metric_type={metric_type}, role={role}, project_type={project_type}")


def create_tagger() -> DashboardPlacementTagger:
    return DashboardPlacementTagger()
