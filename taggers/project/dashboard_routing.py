"""dashboard_routing tagger — which dashboards this survey should appear on.

Stage 4, multi-label, hybrid+LLM. Uses only Stage 1-2 tags + context signals
(per audit F1 fix — must NOT depend on other Stage 4 taggers).
"""

from __future__ import annotations

from models.context import UnifiedContext
from models.tags import TagAccumulator, TagResult
from taggers.base import ProjectTagger


class DashboardRoutingTagger(ProjectTagger):
    name = "project.dashboard_routing"
    tag_dimension = "dashboard_routing"
    stage = 4
    source_type = "hybrid"

    @property
    def depends_on(self) -> list[str]:
        return ["project.project_type", "project.audience"]

    def tag(self, context: UnifiedContext, accumulator: TagAccumulator) -> TagResult:
        project_type = accumulator.get_project_tag_value("project_type")
        audience = accumulator.get_project_tag_value("audience_type") or ""

        dashboards: list[str] = []

        # Executive Dashboard if standard headline metric present
        if context.has_nps or context.has_csat:
            dashboards.append("Executive Dashboard")

        # project_type-based dashboards
        if project_type == "EX":
            dashboards.append("Employee Engagement Dashboard")
            if "Employees" in audience:
                dashboards.append("HR Dashboard")
        elif project_type == "CX":
            dashboards.append("Customer Experience Dashboard")

        # Voice-of-* dashboard if there are open-ended questions
        has_text = any(q.question_type == "T" and not q.is_content_message
                       for q in context.questions)
        if has_text:
            if project_type == "CX":
                dashboards.append("Voice of Customer Dashboard")
            elif project_type == "EX":
                dashboards.append("Voice of Employee Dashboard")

        # Dedupe while preserving insertion order
        seen = set()
        deduped = []
        for d in dashboards:
            if d not in seen:
                deduped.append(d)
                seen.add(d)

        if not deduped:
            # Safety floor: always emit at least one canonical dashboard
            if project_type == "EX":
                deduped = ["Employee Engagement Dashboard"]
            else:
                deduped = ["Customer Experience Dashboard"]

        return TagResult(value=deduped, source="hybrid", confidence=0.70,
                         evidence=f"Rules-based from project_type={project_type}, has_nps={context.has_nps}, has_csat={context.has_csat}")


def create_tagger() -> DashboardRoutingTagger:
    return DashboardRoutingTagger()
